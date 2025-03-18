"""(Batch) Closed World Assumption."""

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader, Dataset

from pykeen.models.nbase import ERModel
from pykeen.training.training_loop import TrainingLoop
from pykeen.triples import CoreTriplesFactory
from pykeen.typing import COLUMN_TAIL, BoolTensor, FloatTensor, LongTensor


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


class BatchCWADataset(Dataset[BatchCWABatch]):
    """A map-style dataset for BCWA training."""

    def __init__(self, mapped_triples: LongTensor):
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


def _scan_triples(other_triples: LongTensor, indices: Sequence[LongTensor]) -> LongTensor:
    """Collect all triples that solely contain the given head/relation/tail indices."""
    mask: BoolTensor = torch.ones(len(other_triples), dtype=torch.bool)
    for i, test_elements in enumerate(indices):
        mask &= torch.isin(elements=other_triples[:, i], test_elements=test_elements)
    return other_triples[mask]


def _convert_to_batch_local(xs: LongTensor) -> tuple[Sequence[LongTensor], LongTensor]:
    """Convert to batch local indices.

    :param xs: shape: (n, d)
        The input tensor.

    :return:
        A tuple (unique, inverse) containing the unique indices per column, and the local tensor.
        The unique indices per column can have different lengths.
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

    .. warning::
        The current implementation is rather memory demanding.
    """

    def __init__(self, mapped_triples: LongTensor):
        """Initialize the collator.

        :param mapped_triples: shape: (num_triples, 3)
            The ID-based training triples.
        """
        self.mapped_triples = mapped_triples

    def __call__(self, batch: list[BatchCWABatch]) -> BatchCWABatch:
        # collect indices
        hs = torch.stack([b.hs for b in batch])
        rs = torch.stack([b.rs for b in batch])
        ts = torch.stack([b.ts for b in batch])

        other_triples = _scan_triples(self.mapped_triples, indices=[hs, rs, ts])
        # batch contains training triples -> we need to find at least those
        assert other_triples.shape[0] >= len(batch)

        # convert to batch local indices
        (hs_uniq, rs_uniq, ts_uniq), targets = _convert_to_batch_local(xs=other_triples)

        return BatchCWABatch(hs=hs_uniq, rs=rs_uniq, ts=ts_uniq, targets=targets)


class BatchCWATrainingLoop(TrainingLoop[LongTensor, BatchCWABatch]):
    """A training loop for BCWA training."""

    def _create_training_data_loader(
        self, triples_factory: CoreTriplesFactory, *, sampler: str | None, batch_size: int, drop_last: bool, **kwargs
    ) -> DataLoader[BatchCWABatch]:
        if sampler:
            raise NotImplementedError("No support for custom samplers yet.")
        mapped_triples = triples_factory.mapped_triples
        return DataLoader(
            dataset=BatchCWADataset(mapped_triples=mapped_triples),
            batch_size=batch_size,
            drop_last=drop_last,
            **kwargs,
            collate_fn=BatchCWACollator(mapped_triples=mapped_triples),
        )

    @staticmethod
    def _get_batch_size(batch: BatchCWABatch) -> int:
        return batch.hs.shape[0]

    def _process_batch(
        self, batch: BatchCWABatch, start: int, stop: int, label_smoothing: float = 0, slice_size: int | None = None
    ) -> FloatTensor:
        if stop - start < self._get_batch_size(batch):
            raise NotImplementedError("No support for default sub-batching.")
        if batch.targets is None:
            raise AssertionError(f"{self} requires a custom collator to fill batch.targets")

        # indices: shape: (num_heads, num_relations, num_tails)
        h_indices = batch.hs.view(-1, 1, 1)
        r_indices = batch.rs.view(1, -1, 1)
        t_indices = batch.ts.view(1, 1, -1)

        # calculate scores
        model = self.model
        assert isinstance(model, ERModel)
        scores: FloatTensor = self.model(
            h_indices=h_indices,
            r_indices=r_indices,
            t_indices=t_indices,
            slice_size=slice_size,
            slice_dim=COLUMN_TAIL,
            mode=None,
        )

        # calculate loss
        return (
            # loss
            self.loss.process_bcwa_scores(scores, targets=batch.targets, label_smoothing=label_smoothing)
            # regularization
            + self.model.collect_regularization_term()
        )

    def _slice_size_search(
        self, *, triples_factory: CoreTriplesFactory, batch_size: int, sub_batch_size: int, supports_sub_batching: bool
    ) -> int:
        raise NotImplementedError("No support for slicing.")
