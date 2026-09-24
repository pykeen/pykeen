"""Training KGE models based on the batch-local closed world assumption (BCWA)."""

from collections.abc import Iterator, Sequence
from math import ceil
from typing import Any, ClassVar, Literal, NamedTuple

import torch
from torch.utils.data import DataLoader, Dataset

from .training_loop import TrainingLoop
from ..models import ERModel
from ..triples import CoreTriplesFactory
from ..triples.instances import SubGraphSLCWAInstances
from ..triples.weights import LossWeighter, loss_weighter_resolver
from ..typing import COLUMN_HEAD, COLUMN_TAIL, FloatTensor, LongTensor, MappedTriples

__all__ = [
    "BatchCWATrainingLoop",
]


class BatchCWABatch(NamedTuple):
    """A batch for BCWA training."""

    hs: LongTensor
    """The unique head entity indices, shape: (num_unique_heads,)."""

    rs: LongTensor
    """The unique relation indices, shape: (num_unique_relations,)."""

    ts: LongTensor
    """The unique tail entity indices, shape: (num_unique_tails,)."""

    targets: LongTensor | None
    """The indices of positive targets, in batch-local indices, shape: (num_positive_triples, 3)

    Only filled during collation.
    """

    weights: FloatTensor | None = None
    """Sample weights, shape: (num_unique_heads, num_unique_relations, num_unique_tails)."""


class BatchCWADataset(Dataset[BatchCWABatch]):
    """A map-style dataset for BCWA training."""

    def __init__(self, mapped_triples: MappedTriples) -> None:
        """Initialize the dataset.

        :param mapped_triples: shape: (num_triples, 3)
            The ID-based training triples.
        """
        super().__init__()
        self.mapped_triples = mapped_triples

    def __getitem__(self, item: int) -> BatchCWABatch:
        h, r, t = self.mapped_triples[item]
        return BatchCWABatch(hs=h, rs=r, ts=t, targets=None)

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

    def __call__(self, batch: list[BatchCWABatch]) -> BatchCWABatch:  # noqa:D102
        # collect indices
        hs = torch.stack([b.hs for b in batch]).unique()
        rs = torch.stack([b.rs for b in batch]).unique()
        ts = torch.stack([b.ts for b in batch]).unique()

        other_triples = self.index.find(hs=hs, rs=rs, ts=ts)
        # batch contains training triples -> we need to find at least those
        assert other_triples.shape[0] >= len(batch)

        # convert to batch local indices
        (hs_uniq, rs_uniq, ts_uniq), targets = _convert_to_batch_local(xs=other_triples)

        weights = None
        if self.loss_weighter is not None:
            shape = (len(hs_uniq), len(rs_uniq), len(ts_uniq))
            weights = self.loss_weighter(h=hs_uniq.view(-1, 1, 1), r=rs_uniq.view(1, -1, 1), t=ts_uniq.view(1, 1, -1))
            # loss weighters may only depend on a subset of h/r/t (e.g. RelationLossWeighter ignores h/t);
            # clone() since broadcast_to returns a non-writable expanded view (e.g. incompatible with pin_memory).
            weights = weights.broadcast_to(shape).clone()

        return BatchCWABatch(hs=hs_uniq, rs=rs_uniq, ts=ts_uniq, targets=targets, weights=weights)


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

    The loss is computed by :meth:`pykeen.losses.Loss.process_bcwa_scores`, which treats each (head, relation)-pair of
    the batch as a row of 1:n scores over the batch's tails.

    The ``batch_size`` refers to the number of sampled training triples. Sub-batching splits along the batch's unique
    heads, and slicing along its unique tails.

    .. note::
        This training loop requires an :class:`pykeen.models.ERModel`.

    Example
    -------
    .. literalinclude:: ../examples/training/bcwa.py
    """

    supports_slicing: ClassVar[bool] = True

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the training loop.

        :param kwargs:
            Keyword-based parameters passed to :meth:`TrainingLoop.__init__`

        :raises TypeError:
            If the model is not an :class:`pykeen.models.ERModel`.
        """
        super().__init__(**kwargs)
        if not isinstance(self.model, ERModel):
            raise TypeError(f"{self.__class__.__name__} requires an ERModel, but got {self.model.__class__.__name__}")

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
        # sub-batching splits along the unique heads
        return batch.hs.shape[0]

    def _process_batch(
        self,
        batch: BatchCWABatch,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: int | None = None,
    ) -> FloatTensor:  # noqa: D102
        if batch.targets is None:
            raise AssertionError(f"{self} requires a custom collator to fill batch.targets")

        # select the sub-batch's heads, and the positive triples with these heads
        targets = batch.targets
        targets = targets[(targets[:, COLUMN_HEAD] >= start) & (targets[:, COLUMN_HEAD] < stop)]
        targets[:, COLUMN_HEAD] -= start
        weights = batch.weights
        if weights is not None:
            weights = weights[start:stop].to(device=self.model.device)

        # calculate scores, shape: (num_heads, num_relations, num_tails)
        device = self.model.device
        scores: FloatTensor = self.model(
            h_indices=batch.hs[start:stop].to(device=device).view(-1, 1, 1),
            r_indices=batch.rs.to(device=device).view(1, -1, 1),
            t_indices=batch.ts.to(device=device).view(1, 1, -1),
            slice_size=slice_size,
            slice_dim=COLUMN_TAIL,
            mode=self.mode,
        )

        return (
            # loss
            self.loss.process_bcwa_scores(
                predictions=scores,
                targets=targets.to(device=device),
                label_smoothing=label_smoothing,
                weights=weights,
            )
            # regularization
            + self.model.collect_regularization_term()
        )

    def _slice_size_search(
        self, *, triples_factory: CoreTriplesFactory, batch_size: int, sub_batch_size: int, supports_sub_batching: bool
    ) -> int:  # noqa: D102
        # slicing is along the batch's unique tails, of which there are at most batch_size
        return self._search_slice_size(
            triples_factory=triples_factory,
            batch_size=batch_size,
            sub_batch_size=sub_batch_size,
            initial_slice_size=ceil(min(batch_size, self.model.num_entities) / 2),
        )
