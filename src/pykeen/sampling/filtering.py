"""Filterer for negative triples."""
from abc import abstractmethod
from typing import Optional, Tuple

import torch
from class_resolver import Resolver
from torch import nn

from ..triples import CoreTriplesFactory


class Filterer(nn.Module):
    """An interface for filtering methods for negative triples."""

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        negative_batch: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, Optional[torch.BoolTensor]]:
        """Filter all proposed negative samples that are positive in the training dataset.

        Normally there is a low probability that proposed negative samples are positive in the training datasets and
        thus act as false negatives. This is expected to act as a kind of regularization, since it adds noise signal to
        the training data. However, the degree of regularization is hard to control since the added noise signal depends
        on the ratio of true triples for a given entity relation or entity entity pair. Therefore, the effects are hard
        to control and a researcher might want to exclude the possibility of having false negatives in the proposed
        negative triples.

        .. note ::
            Filtering is a very expensive task, since every proposed negative sample has to be checked against the
            entire training dataset.

        :param negative_batch: shape: ???
            The batch of negative triples.

        :return:
            A pair (filtered_negative_batch, keep_mask) of shape ???
        """
        raise NotImplementedError


class NoFilterer(Filterer):
    """Dummy filterer which just forwards the batch."""

    def forward(
        self,
        negative_batch: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, Optional[torch.BoolTensor]]:  # noqa: D102
        return negative_batch, None


class DefaultFilterer(Filterer):
    """The default filterer."""

    def __init__(self, triples_factory: CoreTriplesFactory):
        super().__init__()
        # Make sure the mapped triples are initiated
        # Copy the mapped triples to the device for efficient filtering
        self.register_buffer(name="mapped_triples", tensor=triples_factory.mapped_triples)

    def forward(
        self,
        negative_batch: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, Optional[torch.BoolTensor]]:  # noqa: D102
        try:
            # Check which heads of the mapped triples are also in the negative triples
            head_filter = (
                self.mapped_triples[:, 0:1].view(1, -1) == negative_batch[:, 0:1]  # type: ignore
            ).max(axis=0)[0]
            # Reduce the search space by only using possible matches that at least contain the head we look for
            sub_mapped_triples = self.mapped_triples[head_filter]  # type: ignore
            # Check in this subspace which relations of the mapped triples are also in the negative triples
            relation_filter = (sub_mapped_triples[:, 1:2].view(1, -1) == negative_batch[:, 1:2]).max(axis=0)[0]
            # Reduce the search space by only using possible matches that at least contain head and relation we look for
            sub_mapped_triples = sub_mapped_triples[relation_filter]
            # Create a filter indicating which of the proposed negative triples are positive in the training dataset
            final_filter = (sub_mapped_triples[:, 2:3].view(1, -1) == negative_batch[:, 2:3]).max(axis=1)[0]
        except RuntimeError as e:
            # In cases where no triples should be filtered, the subspace reduction technique above will fail
            if str(e) == (
                'cannot perform reduction function max on tensor with no elements because the operation does not '
                'have an identity'
            ):
                final_filter = torch.zeros(negative_batch.shape[0], dtype=torch.bool, device=negative_batch.device)
            else:
                raise e
        # Return only those proposed negative triples that are not positive in the training dataset
        return negative_batch[~final_filter], ~final_filter


filterer_resolver = Resolver.from_subclasses(
    base=Filterer,
    default=DefaultFilterer,
)
