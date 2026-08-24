"""Basic structure for a negative sampler."""

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from class_resolver import HintOrType, normalize_string
from torch import nn

from .filtering import Filterer, filterer_resolver
from ..constants import TARGET_TO_INDEX
from ..typing import BoolTensor, LongTensor, MappedTriples, Target

__all__ = [
    "NegativeSampler",
    "GroupedNegatives",
    "expand_corruption",
]

#: A mapping from corrupted target to the replacement IDs, shape: `(*batch_dims, k_target)`
GroupedNegatives = Mapping[Target, LongTensor]


def expand_corruption(positive_batch: LongTensor, target: Target, replacements: LongTensor) -> LongTensor:
    """Materialise grouped corruptions as dense triples.

    :param positive_batch: shape: `(*batch_dims, 3)` the positive triples.
    :param target: the corrupted target.
    :param replacements: shape: `(*batch_dims, k_target)` the replacement IDs for the corrupted target.

    :returns: shape: `(*batch_dims, k_target, 3)` the dense (materialised) negative triples.
    """
    k_target = replacements.shape[-1]
    negative_batch = positive_batch.unsqueeze(dim=-2).repeat_interleave(k_target, dim=-2)
    negative_batch[..., TARGET_TO_INDEX[target]] = replacements
    return negative_batch


class NegativeSampler(nn.Module):
    """A negative sampler."""

    #: The default strategy for optimizing the negative sampler's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Mapping[str, Any]]] = {
        "num_negs_per_pos": {"type": int, "low": 1, "high": 100, "log": True},
    }

    #: Whether this negative sampler supports :meth:`corrupt_batch_grouped`
    supports_grouped_corruption: ClassVar[bool] = False

    #: A filterer for negative batches
    filterer: Filterer | None

    num_entities: int
    num_relations: int
    num_negs_per_pos: int

    def __init__(
        self,
        *,
        mapped_triples: MappedTriples,
        num_entities: int | None = None,
        num_relations: int | None = None,
        num_negs_per_pos: int | None = None,
        filtered: bool = False,
        filterer: HintOrType[Filterer] = None,
        filterer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the negative sampler with the given entities.

        :param mapped_triples: the positive training triples
        :param num_entities: the number of entities. If None, will be inferred from the triples.
        :param num_relations: the number of relations. If None, will be inferred from the triples.
        :param num_negs_per_pos: number of negative samples to make per positive triple. Defaults to 1.
        :param filtered: Whether proposed corrupted triples that are in the training data should be filtered. Defaults
            to False, since filtering is comparatively expensive and the effect on training is usually small.
        :param filterer: If filtered is set to True, this can be used to choose which filter module from
            :mod:`pykeen.sampling.filtering` is used.
        :param filterer_kwargs: Additional keyword-based arguments passed to the filterer upon construction.
        """
        super().__init__()
        self.num_entities = num_entities or mapped_triples[:, [0, 2]].max().item() + 1
        self.num_relations = num_relations or mapped_triples[:, 1].max().item() + 1
        self.num_negs_per_pos = num_negs_per_pos if num_negs_per_pos is not None else 1
        self.filterer = (
            filterer_resolver.make(
                filterer,
                pos_kwargs=filterer_kwargs,
                mapped_triples=mapped_triples,
            )
            if filterer is not None or filtered
            else None
        )

    @classmethod
    def get_normalized_name(cls) -> str:
        """Get the normalized name of the negative sampler."""
        return normalize_string(cls.__name__, suffix=NegativeSampler.__name__)

    def sample(self, positive_batch: LongTensor) -> tuple[LongTensor, BoolTensor | None]:
        """Generate negative samples from the positive batch.

        :param positive_batch: shape: (batch_size, 3) The positive triples.

        :returns: A pair `(negative_batch, filter_mask)` where

            1. `negative_batch`: shape: (batch_size, num_negatives, 3) The negative batch. ``negative_batch[i, :, :]``
               contains the negative examples generated from ``positive_batch[i, :]``.
            2. filter_mask: shape: (batch_size, num_negatives) An optional filter mask. True where negative samples are
               valid.
        """
        # create unfiltered negative batch by corruption
        negative_batch = self.corrupt_batch(positive_batch=positive_batch)

        if self.filterer is None:
            return negative_batch, None

        # If filtering is activated, all negative triples that are positive in the training dataset will be removed
        return negative_batch, self.filterer(negative_batch=negative_batch)

    @abstractmethod
    def corrupt_batch(self, positive_batch: LongTensor) -> LongTensor:
        """Generate negative samples from the positive batch without application of any filter.

        :param positive_batch: shape: `(*batch_dims, 3)` The positive triples.

        :returns: shape: `(*batch_dims, num_negs_per_pos, 3)` The negative triples. ``result[*bi, :, :]`` contains the
            negative examples generated from ``positive_batch[*bi, :]``.
        """
        raise NotImplementedError

    def corrupt_batch_grouped(self, positive_batch: LongTensor) -> GroupedNegatives:
        """Generate negative samples from the positive batch, grouped by corrupted target.

        Unlike :meth:`corrupt_batch`, this keeps the replacement IDs grouped by which column of the triple they
        replace, which allows scoring each group with a single call to e.g. :meth:`pykeen.models.Model.score_t`
        instead of scoring `num_negs_per_pos` independent triples via :meth:`pykeen.models.Model.score_hrt`.

        :param positive_batch: shape: `(*batch_dims, 3)` The positive triples.

        :returns: a mapping from corrupted target to the replacement IDs, shape: `(*batch_dims, k_target)`, with
            ``sum(k_target for k_target in ...) == num_negs_per_pos``.

        :raises NotImplementedError: if this sampler does not support grouped corruption, cf.
            :data:`supports_grouped_corruption`.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support grouped corruption.")

    def sample_grouped(self, positive_batch: LongTensor) -> tuple[GroupedNegatives, Mapping[Target, BoolTensor] | None]:
        """Generate grouped negative samples from the positive batch, filtering out known positives if configured.

        :param positive_batch: shape: `(*batch_dims, 3)` The positive triples.

        :returns: A pair `(grouped_negatives, filter_masks)` where

            1. `grouped_negatives`: a mapping from corrupted target to the replacement IDs, shape:
               `(*batch_dims, k_target)`.
            2. `filter_masks`: an optional mapping from corrupted target to a filter mask, shape:
               `(*batch_dims, k_target)`. True where the corresponding negative sample is valid.
        """
        grouped_negatives = self.corrupt_batch_grouped(positive_batch=positive_batch)

        if self.filterer is None:
            return grouped_negatives, None

        # materialise dense triples per target only to feed the filterer; no representations involved
        filter_masks = {
            target: self.filterer(
                negative_batch=expand_corruption(
                    positive_batch=positive_batch, target=target, replacements=replacements
                )
            )
            for target, replacements in grouped_negatives.items()
        }
        return grouped_negatives, filter_masks
