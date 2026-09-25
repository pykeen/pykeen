"""Training KGE models based on the batch-local closed world assumption (BCWA)."""

from collections.abc import Iterator, Sequence
from math import ceil
from typing import Any, ClassVar, Literal

import torch
from torch.utils.data import DataLoader, Dataset

from .training_loop import TrainingLoop
from ..constants import get_target_column
from ..models import ERModel
from ..triples import CoreTriplesFactory
from ..triples.instances import BatchCWABatch, SubGraphSLCWAInstances
from ..triples.weights import LossWeighter, loss_weighter_resolver
from ..typing import COLUMN_HEAD, COLUMN_RELATION, FloatTensor, LongTensor, MappedTriples, TargetHint

__all__ = [
    "BatchCWATrainingLoop",
    "MissingBatchTargetsError",
    "UnknownBatchTriplesError",
]


class MissingBatchTargetsError(ValueError):
    """Raised if a BCWA batch does not contain the positive triples, i.e., was not created by the BCWA collator."""


class UnknownBatchTriplesError(ValueError):
    """Raised if a batch contains triples which are not part of the triples the BCWA collator was created for."""


class BatchCWADataset(Dataset[LongTensor]):
    """A map-style dataset of single triples for BCWA training, which are collated by :class:`BatchCWACollator`."""

    def __init__(self, mapped_triples: MappedTriples) -> None:
        """Initialize the dataset.

        :param mapped_triples: shape: (num_triples, 3)
            The ID-based training triples.
        """
        super().__init__()
        self.mapped_triples = mapped_triples

    def __getitem__(self, item: int) -> LongTensor:
        return self.mapped_triples[item]

    def __len__(self) -> int:
        return self.mapped_triples.shape[0]


class _HeadIndex:
    """An index of triples by their head entity."""

    def __init__(self, mapped_triples: MappedTriples) -> None:
        """Initialize the index.

        :param mapped_triples: shape: (num_triples, 3)
            The ID-based triples.
        """
        heads = mapped_triples[:, COLUMN_HEAD]
        self.mapped_triples = mapped_triples[heads.argsort()]
        degree = torch.bincount(heads)
        self.offsets = torch.cat([degree.new_zeros(1), degree.cumsum(dim=0)])

    def find(self, hs: LongTensor, rs: LongTensor, ts: LongTensor) -> MappedTriples:
        """Find all triples which solely consist of the given heads, relations, and tails.

        :param hs: shape: (num_heads,)
            The unique head indices.
        :param rs: shape: (num_relations,)
            The relation indices.
        :param ts: shape: (num_tails,)
            The tail indices.

        :return: shape: (num_found_triples, 3)
            The triples.
        """
        # heads beyond the indexed ones do not have any triples
        hs = hs[hs < len(self.offsets) - 1]
        # gather all triples with a matching head; the cost is proportional to the total degree of the heads
        starts = self.offsets[hs]
        counts = self.offsets[hs + 1] - starts
        index = torch.repeat_interleave(starts - (counts.cumsum(dim=0) - counts), counts) + torch.arange(
            int(counts.sum())
        )
        candidates = self.mapped_triples[index]
        # filter by relation and tail
        mask = torch.isin(candidates[:, 1], rs) & torch.isin(candidates[:, 2], ts)
        return candidates[mask]


def _convert_to_batch_local(xs: LongTensor) -> tuple[Sequence[LongTensor], LongTensor]:
    """Convert to batch local indices.

    :param xs: shape: (n, d)
        The input tensor.

    :return:
        A tuple (unique, inverse) containing the unique indices per column, and the local tensor.
        The unique indices per column can have different lengths.

    :raises ValueError:
        If the input is not two-dimensional.
    """
    if xs.ndimension() != 2:
        raise ValueError(f"Invalid shape: {xs.shape=}")
    uniqs = []
    targets = []
    for dim in range(xs.shape[1]):
        uniq, inv = xs[:, dim].unique(return_inverse=True)
        uniqs.append(uniq)
        targets.append(inv)
    return uniqs, torch.stack(targets, dim=-1)


class BatchCWACollator:
    """A custom collator for BCWA training.

    It collects all training triples whose head, relation, and tail each occur in the batch.
    """

    def __init__(self, mapped_triples: MappedTriples, loss_weighter: LossWeighter | None = None) -> None:
        """Initialize the collator.

        :param mapped_triples: shape: (num_triples, 3)
            The ID-based training triples.
        :param loss_weighter:
            The method to determine sample weights.
        """
        self.index = _HeadIndex(mapped_triples=mapped_triples)
        self.loss_weighter = loss_weighter

    def __call__(self, batch: list[LongTensor]) -> BatchCWABatch:
        """Collate a batch of single triples.

        :param batch: The single triples.

        :returns: The collated batch, with the unique IDs per position, the positive targets, and optional weights.

        :raises UnknownBatchTriplesError: If the batch contains triples which the collator does not know.
        """
        # collect indices
        hs, rs, ts = (column.unique() for column in torch.stack(batch).unbind(dim=-1))

        other_triples = self.index.find(hs=hs, rs=rs, ts=ts)
        # the batch consists of training triples -> we need to find at least those
        if other_triples.shape[0] < len(batch):
            raise UnknownBatchTriplesError(
                f"Found only {other_triples.shape[0]} triples for a batch of {len(batch)} triples. The batch has to be "
                f"drawn from the same triples the collator was created for."
            )

        # convert to batch local indices
        (hs_uniq, rs_uniq, ts_uniq), positives = _convert_to_batch_local(xs=other_triples)
        result = BatchCWABatch(heads=hs_uniq, relations=rs_uniq, tails=ts_uniq, positives=positives)

        if self.loss_weighter is not None:
            shape = (len(hs_uniq), len(rs_uniq), len(ts_uniq))
            weights = self.loss_weighter(h=hs_uniq.view(-1, 1, 1), r=rs_uniq.view(1, -1, 1), t=ts_uniq.view(1, 1, -1))
            # loss weighters may only depend on a subset of h/r/t (e.g. RelationLossWeighter ignores h/t);
            # clone() since broadcast_to returns a non-writable expanded view (e.g. incompatible with pin_memory).
            result["weights"] = weights.broadcast_to(shape).clone()

        return result


class _SubGraphBatchSampler:
    """A batch sampler yielding the triple IDs of coherent subgraphs, cf. :class:`SubGraphSLCWAInstances`."""

    def __init__(self, instances: SubGraphSLCWAInstances) -> None:
        self.instances = instances

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.instances.iter_triple_ids())

    def __len__(self) -> int:
        return len(self.instances)


class BatchCWATrainingLoop(TrainingLoop[BatchCWABatch]):
    r"""A training loop that is based upon the batch-local closed world assumption (BCWA).

    For a batch of training triples, it collects the sets of heads $\mathcal{H}_B$, relations $\mathcal{R}_B$, and
    tails $\mathcal{T}_B$ occurring in it, and scores all triples in $\mathcal{H}_B \times \mathcal{R}_B \times
    \mathcal{T}_B$ at once. All of these triples which are training triples are considered as positive, and all others
    as negative. This makes better use of the calculated representations, similar to the in-batch negatives commonly
    used in contrastive learning (e.g., [zhai2023]_).

    The loss is computed by :meth:`~pykeen.losses.Loss.process_bcwa_scores`, which treats each combination of the two
    non-target positions as a row of 1:n scores over the batch's candidates for the target position. Like for
    :class:`~pykeen.training.LCWATrainingLoop`, the target defaults to the tail.

    The ``batch_size`` refers to the number of sampled training triples. Since these are turned into the grid of all
    combinations, there is no batch dimension left to split, and thus, sub-batching is not supported. To reduce the
    memory requirements, slicing splits the score computation along the target position.

    .. todo::
        The loss decomposes over the combinations of the two non-target positions. Thus, the grid could be split along
        these two positions, as long as each part keeps all target candidates, and the loss is normalized by the number
        of rows in the full batch.

    .. note::
        This training loop requires an :class:`~pykeen.models.ERModel`.

    Example
    -------
    .. literalinclude:: ../examples/training/bcwa.py
    """

    supports_slicing: ClassVar[bool] = True
    supports_sub_batching: ClassVar[bool] = False

    def __init__(self, *, target: TargetHint = None, **kwargs: Any) -> None:
        """
        Initialize the training loop.

        :param target:
            The target column. Defaults to tail prediction.
        :param kwargs:
            Keyword-based parameters passed to :meth:`TrainingLoop.__init__`

        :raises TypeError:
            If the model is not an :class:`~pykeen.models.ERModel`.
        """
        super().__init__(**kwargs)
        if not isinstance(self.model, ERModel):
            raise TypeError(f"{self.__class__.__name__} requires an ERModel, but got {self.model.__class__.__name__}")
        self.target = get_target_column(target)
        # the order of dimensions which moves the target dimension last
        self._order = [dim for dim in range(3) if dim != self.target] + [self.target]

    def _create_training_data_loader(
        self,
        triples_factory: CoreTriplesFactory,
        *,
        sampler: Literal["schlichtkrull"] | None,
        batch_size: int,
        drop_last: bool,
        **kwargs: Any,
    ) -> DataLoader[BatchCWABatch]:  # noqa: D102
        mapped_triples = triples_factory._add_inverse_triples_if_necessary(
            mapped_triples=triples_factory.mapped_triples
        )
        dataset = BatchCWADataset(mapped_triples=mapped_triples)
        collate_fn = BatchCWACollator(
            mapped_triples=mapped_triples,
            loss_weighter=loss_weighter_resolver.make_safe(self.loss_weighter, self.loss_weighter_kwargs),
        )
        match sampler:
            case None:
                return DataLoader(
                    dataset=dataset, batch_size=batch_size, drop_last=drop_last, collate_fn=collate_fn, **kwargs
                )
            case "schlichtkrull":
                # the sub-graph sampler determines the order of the triples itself
                kwargs.pop("shuffle", None)
                instances = SubGraphSLCWAInstances(
                    mapped_triples=mapped_triples,
                    batch_size=batch_size,
                    drop_last=drop_last,
                    num_entities=triples_factory.num_entities,
                    num_relations=triples_factory.num_relations,
                )
                return DataLoader(
                    dataset=dataset,
                    batch_sampler=_SubGraphBatchSampler(instances=instances),
                    collate_fn=collate_fn,
                    **kwargs,
                )
            case _:
                raise ValueError(f"Invalid {sampler=}")

    @staticmethod
    def _get_batch_size(batch: BatchCWABatch) -> int:  # noqa: D102
        # the batch cannot be split, cf. supports_sub_batching
        return 1

    def _process_batch(
        self,
        batch: BatchCWABatch,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: int | None = None,
    ) -> FloatTensor:  # noqa: D102
        if "positives" not in batch:
            raise MissingBatchTargetsError(
                f"{self.__class__.__name__} requires batches with positives, as created by {BatchCWACollator.__name__}."
            )

        positives = batch["positives"]
        weights = batch.get("weights")

        # calculate scores for all combinations, shape: (num_heads, num_relations, num_tails)
        device = self.model.device
        scores: FloatTensor = self.model(
            h_indices=batch["heads"].to(device=device).view(-1, 1, 1),
            r_indices=batch["relations"].to(device=device).view(1, -1, 1),
            t_indices=batch["tails"].to(device=device).view(1, 1, -1),
            slice_size=slice_size,
            slice_dim=self.target,
            mode=self.mode,
        )

        # move the target dimension last
        scores = scores.permute(*self._order)
        positives = positives[:, self._order]
        if weights is not None:
            weights = weights.permute(*self._order).to(device=device)

        return (
            # loss
            self.loss.process_bcwa_scores(
                predictions=scores,
                positives=positives.to(device=device),
                label_smoothing=label_smoothing,
                weights=weights,
            )
            # regularization
            + self.model.collect_regularization_term()
        )

    def _get_initial_slice_size(self, batch_size: int) -> int:  # noqa: D102
        # slicing is along the batch's unique targets, of which there are at most batch_size
        num_targets = self.model.num_relations if self.target == COLUMN_RELATION else self.model.num_entities
        return ceil(min(batch_size, num_targets) / 2)
