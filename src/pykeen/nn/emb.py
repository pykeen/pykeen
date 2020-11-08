# -*- coding: utf-8 -*-

"""Embedding modules."""

import functools
from typing import Any, Mapping, Optional

import torch
import torch.nn
from torch import nn

__all__ = [
    'RepresentationModule',
    'Embedding',
]

from pykeen.typing import Constrainer, Initializer, Normalizer


class RepresentationModule(nn.Module):
    """A base class for obtaining representations for entities/relations."""

    def forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """Get representations for indices.

        :param indices: shape: (m,)
            The indices, or None. If None, return all representations.

        :return: shape: (m, d)
            The representations.
        """
        raise NotImplementedError

    def reset_parameters(self) -> None:
        """Reset the module's parameters."""

    def post_parameter_update(self):
        """Apply constraints which should not be included in gradients."""


class Embedding(RepresentationModule):
    """Trainable embeddings.

    This class provides the same interface as :class:`torch.nn.Embedding` and
    can be used throughout PyKEEN as a more fully featured drop-in replacement.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        initializer: Optional[Initializer] = None,
        initializer_kwargs: Optional[Mapping[str, Any]] = None,
        normalizer: Optional[Normalizer] = None,
        normalizer_kwargs: Optional[Mapping[str, Any]] = None,
        constrainer: Optional[Constrainer] = None,
        constrainer_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """Instantiate an embedding with extended functionality.

        :param num_embeddings: >0
            The number of embeddings.
        :param embedding_dim: >0
            The embedding dimensionality.
        :param initializer:
            An optional initializer, which takes an uninitialized (num_embeddings, embedding_dim) tensor as input,
            and returns an initialized tensor of same shape and dtype (which may be the same, i.e. the
            initialization may be in-place)
        :param initializer_kwargs:
            Additional keyword arguments passed to the initializer
        :param normalizer:
            A normalization function, which is applied in every forward pass.
        :param normalizer_kwargs:
            Additional keyword arguments passed to the normalizer
        :param constrainer:
            A function which is applied to the weights after each parameter update, without tracking gradients.
            It may be used to enforce model constraints outside of gradient-based training. The function does not need
            to be in-place, but the weight tensor is modified in-place.
        :param constrainer_kwargs:
            Additional keyword arguments passed to the constrainer
        """
        super().__init__()

        if initializer is None:
            initializer = nn.init.normal_
        if initializer_kwargs:
            self.initializer = functools.partial(initializer, **initializer_kwargs)
        else:
            self.initializer = initializer
        if constrainer_kwargs:
            self.constrainer = functools.partial(constrainer, **constrainer_kwargs)
        else:
            self.constrainer = constrainer
        if normalizer_kwargs:
            self.normalizer = functools.partial(normalizer, **normalizer_kwargs)
        else:
            self.normalizer = normalizer
        self._embeddings = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

    @classmethod
    def init_with_device(
        cls,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device,
        initializer: Optional[Initializer] = None,
        initializer_kwargs: Optional[Mapping[str, Any]] = None,
        normalizer: Optional[Normalizer] = None,
        normalizer_kwargs: Optional[Mapping[str, Any]] = None,
        constrainer: Optional[Constrainer] = None,
        constrainer_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> 'Embedding':  # noqa:E501
        """Create an embedding object on the given device by wrapping :func:`__init__`.

        This method is a hotfix for not being able to pass a device during initialization of
        :class:`torch.nn.Embedding`. Instead the weight is always initialized on CPU and has
        to be moved to GPU afterwards.

        .. seealso::

            https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application

        :return:
            The embedding.
        """
        return cls(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            initializer=initializer,
            initializer_kwargs=initializer_kwargs,
            normalizer=normalizer,
            normalizer_kwargs=normalizer_kwargs,
            constrainer=constrainer,
            constrainer_kwargs=constrainer_kwargs,
        ).to(device=device)

    @property
    def num_embeddings(self) -> int:  # noqa: D401
        """The total number of representations (i.e. the maximum ID)."""
        return self._embeddings.num_embeddings

    @property
    def embedding_dim(self) -> int:  # noqa: D401
        """The representation dimension."""
        return self._embeddings.embedding_dim

    def reset_parameters(self) -> None:  # noqa: D102
        # initialize weights in-place
        self._embeddings.weight.data = self.initializer(self._embeddings.weight.data)

    def post_parameter_update(self):  # noqa: D102
        # apply constraints in-place
        if self.constrainer is not None:
            self._embeddings.weight.data = self.constrainer(self._embeddings.weight.data)

    def forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        if indices is None:
            x = self._embeddings.weight
        else:
            x = self._embeddings(indices)
        if self.normalizer is not None:
            x = self.normalizer(x)
        return x

    def get_in_canonical_shape(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """Get embedding in canonical shape.

        :param indices: The indices. If None, return all embeddings.

        :return: shape: (batch_size, num_embeddings, d)
        """
        x = self(indices=indices)
        if indices is None:
            return x.unsqueeze(dim=0)
        return x.unsqueeze(dim=1)
