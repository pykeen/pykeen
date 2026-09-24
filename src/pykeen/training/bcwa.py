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
from ..typing import COLUMN_HEAD, COLUMN_RELATION, COLUMN_TAIL, FloatTensor, LongTensor, MappedTriples, TargetHint

__all__ = [
    "BatchCWATrainingLoop",
]


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

    The loss is computed by :meth:`~pykeen.losses.Loss.process_bcwa_scores`, which treats each combination of the two
    non-target positions as a row of 1:n scores over the batch's candidates for the target position. Like for
    :class:`~pykeen.training.LCWATrainingLoop`, the target defaults to the tail.

    The ``batch_size`` refers to the number of sampled training triples. Sub-batching splits along the batch's unique
    heads (or tails, if the heads are the target), and slicing along the target position.

    .. note::
        This training loop requires an :class:`~pykeen.models.ERModel`.

    Example
    -------
    .. literalinclude:: ../examples/training/bcwa.py
    """

    supports_slicing: ClassVar[bool] = True

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
        # the dimension along which to sub-batch
        self.sub_batch_dim = COLUMN_TAIL if self.target == COLUMN_HEAD else COLUMN_HEAD
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

    def _get_batch_size(self, batch: BatchCWABatch) -> int:  # type: ignore[override] # noqa: D102
        return batch[self.sub_batch_dim].shape[0]

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

        # select the sub-batch, and the positive triples within it
        dim = self.sub_batch_dim
        ids = [batch.hs, batch.rs, batch.ts]
        ids[dim] = ids[dim][start:stop]
        targets = batch.targets
        targets = targets[(targets[:, dim] >= start) & (targets[:, dim] < stop)]
        targets[:, dim] -= start
        weights = batch.weights
        if weights is not None:
            weights = weights.narrow(dim, start, stop - start)

        # calculate scores, shape: (num_heads, num_relations, num_tails)
        device = self.model.device
        h_indices, r_indices, t_indices = (
            x.to(device=device).view(*(-1 if i == j else 1 for j in range(3))) for i, x in enumerate(ids)
        )
        scores: FloatTensor = self.model(
            h_indices=h_indices,
            r_indices=r_indices,
            t_indices=t_indices,
            slice_size=slice_size,
            slice_dim=self.target,
            mode=self.mode,
        )

        # move the target dimension last
        scores = scores.permute(*self._order)
        targets = targets[:, self._order]
        if weights is not None:
            weights = weights.permute(*self._order).to(device=device)

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

    def _get_initial_slice_size(self, batch_size: int) -> int:  # noqa: D102
        # slicing is along the batch's unique targets, of which there are at most batch_size
        num_targets = self.model.num_relations if self.target == COLUMN_RELATION else self.model.num_entities
        return ceil(min(batch_size, num_targets) / 2)
