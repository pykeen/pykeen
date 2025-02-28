"""(Batch) Closed World Assumption."""

from typing import NamedTuple

import torch
from torch.utils.data import DataLoader, Dataset

from pykeen.models.nbase import ERModel
from pykeen.training.training_loop import TrainingLoop
from pykeen.triples import CoreTriplesFactory
from pykeen.typing import COLUMN_TAIL, FloatTensor, LongTensor


class BatchCWABatch(NamedTuple):
    # shape: (batch_size,)
    hs: LongTensor
    # shape: (batch_size,)
    rs: LongTensor
    # shape: (batch_size,)
    ts: LongTensor
    # shape: (nnz,), in batch-local indices
    # should contain all triples whose h, r, t indices occur within the batch
    h_target: LongTensor
    r_target: LongTensor
    t_target: LongTensor


class BatchCWADataset(Dataset[BatchCWABatch]):
    def __init__(self, mapped_triples: LongTensor):
        super().__init__()
        self.mapped_triples = mapped_triples

    def __getitem__(self, item: int) -> BatchCWABatch:
        h, r, t = self.mapped_triples[item]
        x = torch.zeros(1, dtype=torch.long)
        return BatchCWABatch(hs=h, rs=r, ts=t, h_target=x, r_target=x, t_target=x)

    def __len__(self) -> int:
        return self.mapped_triples.shape[0]


class BatchCWACollator:
    # note: the implement is relatively memory demanding
    def __init__(self, mapped_triples: LongTensor):
        self.mapped_triples = mapped_triples

    def __call__(self, batch: list[BatchCWABatch]) -> BatchCWABatch:
        # collect
        hs = torch.stack([b.hs for b in batch]).unique()
        rs = torch.stack([b.rs for b in batch]).unique()
        ts = torch.stack([b.ts for b in batch]).unique()

        # collect all triples that solely contain the current batch entities/relations
        other_triples = self.mapped_triples
        for i, indices in enumerate([hs, rs, ts]):
            mask = torch.isin(elements=other_triples[:, i], test_elements=indices)
            other_triples = other_triples[mask]

        # convert to batch local indices
        targets = []
        for dim, indices in enumerate([hs, rs, ts]):
            uniq, inv = other_triples[:, dim].unique(return_inverse=True)
            assert torch.equal(uniq, indices)
            targets.append(inv)

        return BatchCWABatch(hs=hs, rs=rs, ts=ts, h_target=targets[0], r_target=targets[1], t_target=targets[2])


class BatchCWATrainingLoop(TrainingLoop[BatchCWABatch, BatchCWABatch]):
    def _create_training_data_loader(
        self, triples_factory: CoreTriplesFactory, *, sampler: str | None, batch_size: int, drop_last: bool, **kwargs
    ) -> DataLoader[BatchCWABatch]:
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
            self.loss.process_cwa_scores(
                scores, hs=batch.h_target, rs=batch.r_target, ts=batch.t_target, label_smoothing=label_smoothing
            )
            # regularization
            + self.model.collect_regularization_term()
        )

    def _slice_size_search(
        self, *, triples_factory: CoreTriplesFactory, batch_size: int, sub_batch_size: int, supports_sub_batching: bool
    ) -> int:
        raise NotImplementedError("No support for slicing.")
