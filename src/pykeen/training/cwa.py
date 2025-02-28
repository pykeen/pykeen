"""(Batch) Closed World Assumption."""

from typing import NamedTuple

from torch.utils.data import DataLoader

from pykeen.models.nbase import ERModel
from pykeen.training.training_loop import TrainingLoop
from pykeen.triples import CoreTriplesFactory
from pykeen.typing import FloatTensor, LongTensor


class CWABatch(NamedTuple):
    # shape: (batch_size, 3)
    hrt_batch: LongTensor
    # shape: (nnz, 3), in batch-local indices
    hrt_target: LongTensor


class CWATrainingLoop(TrainingLoop[CWABatch, CWABatch]):
    def _create_training_data_loader(
        self, triples_factory: CoreTriplesFactory, *, sampler: str | None, batch_size: int, drop_last: bool, **kwargs
    ) -> DataLoader[CWABatch]:
        raise NotImplementedError

    @staticmethod
    def _get_batch_size(batch: CWABatch) -> int:
        return batch.hrt_batch.shape[0]

    def _process_batch(
        self, batch: CWABatch, start: int, stop: int, label_smoothing: float = 0, slice_size: int | None = None
    ) -> FloatTensor:
        if stop - start < self._get_batch_size(batch):
            raise NotImplementedError("No support for default sub-batching.")

        # indices: shape: (num_heads, num_relations, num_tails)
        h_indices, r_indices, t_indices = batch.hrt_batch.unbind(dim=-1)
        h_indices = h_indices.view(-1, 1, 1)
        r_indices = h_indices.view(1, -1, 1)
        t_indices = h_indices.view(1, 1, -1)

        # calculate scores
        model = self.model
        assert isinstance(model, ERModel)
        scores: FloatTensor = self.model(
            h_indices=h_indices,
            r_indices=r_indices,
            t_indices=t_indices,
            slice_size=slice_size,
            slice_dim=9999,  # TODO
            mode=None,
        )

        # calculate loss
        return (
            # loss
            self.loss.process_cwa_scores(scores, non_zero_indices=batch.hrt_target, label_smoothing=label_smoothing)
            # regularization
            + self.model.collect_regularization_term()
        )

    def _slice_size_search(
        self, *, triples_factory: CoreTriplesFactory, batch_size: int, sub_batch_size: int, supports_sub_batching: bool
    ) -> int:
        raise NotImplementedError("No support for slicing.")
