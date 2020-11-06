# -*- coding: utf-8 -*-

"""Embedding modules."""

import functools
import warnings
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn
from torch import nn

__all__ = [
    'RepresentationModule',
    'Embedding',
    'Initializer',
    'Normalizer',
    'Constrainer',
]


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

    @torch.no_grad()
    def post_parameter_update(self):
        """Apply constraints which should not be included in gradients."""


Initializer = Callable[[nn.Parameter], None]
Normalizer = Callable[[torch.FloatTensor], torch.FloatTensor]
Constrainer = Callable[[torch.FloatTensor], torch.FloatTensor]


class Embedding(RepresentationModule):
    """Trainable embeddings.

    This class provides the same interface as :class:`torch.nn.Embedding` and
    can be used throughout PyKEEN as a more fully featured drop-in replacement.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        initialization: Optional[Initializer] = None,
        initialization_kwargs: Optional[Mapping[str, Any]] = None,
        normalization: Optional[Normalizer] = None,
        constrainer: Optional[Constrainer] = None,
    ):
        super().__init__()

        if initialization is None:
            initialization = nn.init.normal_
        if initialization_kwargs:
            self.initialization = functools.partial(initialization, **initialization_kwargs)
        else:
            self.initialization = initialization
        self.constrainer = constrainer
        self.normalization = normalization
        self._embeddings = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

    @torch.no_grad()
    def post_parameter_update(self):  # noqa: D102
        if self.constrainer is not None:
            self._embeddings.weight.data = self.constrainer(self._embeddings.weight.data)

    @classmethod
    def init_with_device(
        cls,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device,
        initializer: Optional[Initializer] = None,
        initializer_kwargs: Optional[Mapping[str, Any]] = None,
        normalizer: Optional[Normalizer] = None,
        constrainer: Optional[Constrainer] = None,
    ) -> 'Embedding':  # noqa:E501
        """Create an embedding object on a device.

        This method is a hotfix for not being able to pass a device during initialization of
        :class:torch.nn.Embedding`. Instead the weight is always initialized on CPU and has
        to be moved to GPU afterwards.

        .. seealso::

            https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application

        :param num_embeddings: >0
            The number of embeddings.
        :param embedding_dim: >0
            The embedding dimensionality.
        :param device:
            The device.
        :param initializer:
            An optional initializer, which takes a (num_embeddings, embedding_dim) tensor as input, and modifies
            the weights in-place.
        :param initializer_kwargs:
            Additional keyword arguments passed to the initializer
        :param normalizer:
            A normalization function
        :param constrainer:
            A contrainer applied after each parameter update, without tracking gradients.

        :return:
            The embedding.
        """
        return cls(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            initialization=initializer,
            initialization_kwargs=initializer_kwargs,
            normalization=normalizer,
            constrainer=constrainer,
        ).to(device=device)

    @property
    def num_embeddings(self) -> int:  # noqa: D401
        """The total number of representations (i.e. the maximum ID)."""
        return self._embeddings.num_embeddings

    @property
    def embedding_dim(self) -> int:  # noqa: D401
        """The representation dimension."""
        return self._embeddings.embedding_dim

    @property
    def weight(self):  # noqa: D102
        warnings.warn(
            f"{self.__class__.__name__}.weight is deprecated. Use {self.__class__.__name__}(indices=None) instead.",
            DeprecationWarning,
        )
        return self.forward(indices=None)

    def reset_parameters(self) -> None:  # noqa: D102
        self.initialization(self._embeddings.weight)

    def forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        if indices is None:
            x = self._embeddings.weight
        else:
            x = self._embeddings(indices)
        if self.normalization is not None:
            x = self.normalization(x)
        return x
