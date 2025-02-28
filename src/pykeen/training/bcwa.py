"""(Batch) Closed World Assumption."""

from typing import NamedTuple

import torch
from torch.utils.data import DataLoader, Dataset

from pykeen.models.nbase import ERModel
from pykeen.training.training_loop import TrainingLoop
from pykeen.triples import CoreTriplesFactory
from pykeen.typing import COLUMN_TAIL, FloatTensor, LongTensor


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
        # collect unique indices
        hs = torch.stack([b.hs for b in batch]).unique()
        rs = torch.stack([b.rs for b in batch]).unique()
        ts = torch.stack([b.ts for b in batch]).unique()

        # collect all triples that solely contain the current batch heads/relations/tails
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

        return BatchCWABatch(hs=hs, rs=rs, ts=ts, targets=torch.stack(targets, dim=-1))


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
