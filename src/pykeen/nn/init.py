# -*- coding: utf-8 -*-

"""Embedding weight initialization routines."""

import logging
import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn
import torch.nn.init
from torch.nn import functional

from .utils import TransformerEncoder
from ..triples.triples_factory import TriplesFactory
from ..typing import Initializer
from ..utils import compose

__all__ = [
    "create_init_from_pretrained",
    "xavier_uniform_",
    "xavier_uniform_norm_",
    "xavier_normal_",
    "xavier_normal_norm_",
    "uniform_norm_",
    "normal_norm_",
    "init_phases",
]

logger = logging.getLogger(__name__)


def xavier_uniform_(tensor, gain: float = 1.0):
    r"""Initialize weights of the tensor similarly to Glorot/Xavier initialization.

    Proceed as if it was a linear layer with fan_in of zero and Xavier uniform
    initialization is used, i.e. fill the weight of input `embedding` with values values
    sampled from :math:`\mathcal{U}(-a, a)` where

    .. math::

        a = \text{gain} \times \sqrt{\frac{6}{\text{embedding_dim}}}

    :param tensor: A tensor
    :param gain: An optional scaling factor, defaults to 1.0.
    :return: Embedding with weights by the Xavier uniform initializer.
    """
    bound = gain * 6 / math.sqrt(tensor.shape[-1])
    torch.nn.init.uniform_(tensor, -bound, bound)
    return tensor


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    r"""Initialize weights of the tensor similarly to Glorot/Xavier initialization.

    Proceed as if it was a linear layer with fan_in of zero and Xavier normal
    initialization is used. Fill the weight of input `embedding` with values values
    sampled from :math:`\mathcal{N}(0, a^2)` where

    .. math::

        a = \text{gain} \times \sqrt{\frac{2}{\text{embedding_dim}}}

    :param tensor: A tensor
    :param gain: An optional scaling factor, defaults to 1.0.
    :return: Embedding with weights by the Xavier normal initializer.
    """
    std = gain * 2 / math.sqrt(tensor.shape[-1])
    torch.nn.init.normal_(tensor, mean=0.0, std=std)
    return tensor


def init_phases(x: torch.Tensor) -> torch.Tensor:
    r"""Generate random phases between 0 and :math:`2\pi`."""
    phases = 2 * np.pi * torch.rand_like(x[..., : x.shape[-1] // 2])
    return torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1).detach()


xavier_uniform_norm_ = compose(
    torch.nn.init.xavier_uniform_,
    functional.normalize,
)
xavier_normal_norm_ = compose(
    torch.nn.init.xavier_normal_,
    functional.normalize,
)
uniform_norm_ = compose(
    torch.nn.init.uniform_,
    functional.normalize,
)
normal_norm_ = compose(
    torch.nn.init.normal_,
    functional.normalize,
)


def init_quaternions(
    x: torch.FloatTensor,
) -> torch.FloatTensor:
    """Initialize quaternion."""
    num_elements, dim = x.shape
    if dim % 4 != 0:
        raise ValueError(f"Quaternions have four components, but dimension {dim} is not divisible by four.")
    dim //= 4
    # scaling factor
    s = 1.0 / math.sqrt(2 * num_elements)
    # modulus ~ Uniform[-s, s]
    modulus = 2 * s * torch.rand(num_elements, dim) - s
    # phase ~ Uniform[0, 2*pi]
    phase = 2 * math.pi * torch.rand(num_elements, dim)
    # real part
    real = (modulus * phase.cos()).unsqueeze(dim=-1)
    # purely imaginary quaternions unitary
    imag = torch.rand(num_elements, dim, 3)
    imag = functional.normalize(imag, p=2, dim=-1)
    imag = imag * (modulus * phase.sin()).unsqueeze(dim=-1)
    x = torch.cat([real, imag], dim=-1)
    return x.view(num_elements, 4 * dim)


def create_init_from_pretrained(pretrained: torch.FloatTensor) -> Initializer:
    """
    Create an initializer via a constant vector.

    :param pretrained:
        the tensor of pretrained embeddings.

    :return:
        an initializer, which fills a tensor with the given weights.

    Added in https://github.com/pykeen/pykeen/pull/638.

    Example usage:

    .. code-block::

        import torch
        from pykeen.pipeline import pipeline
        from pykeen.nn.init import create_init_from_pretrained

        # this is usually loaded from somewhere else
        # the shape must match, as well as the entity-to-id mapping
        pretrained_embedding_tensor = torch.rand(14, 128)

        result = pipeline(
            dataset="nations",
            model="transe",
            model_kwargs=dict(
                embedding_dim=pretrained_embedding_tensor.shape[-1],
                entity_initializer=create_init_from_pretrained(pretrained_embedding_tensor),
            ),
        )
    """

    def init_from_pretrained(x: torch.FloatTensor) -> torch.FloatTensor:
        """Initialize tensor with pretrained weights."""
        if x.shape != pretrained.shape:
            raise ValueError(f"shape of pretrained {pretrained.shape} does not match shape of tensor {x.shape}")
        return pretrained

    return init_from_pretrained


class LabelBasedInitializer:
    """An initializer using pretrained models from the `transformers` library to encode labels."""

    def __init__(
        self,
        labels: Sequence[str],
        pretrained_model_name_or_path: str = "bert-base-cased",
        batch_size: int = 32,
        max_length: Optional[int] = None,
    ):
        """
        Initialize the initializer.

        :param labels:
            the labels
        :param pretrained_model_name_or_path:
            the name of the pretrained model, or a path, cf. :func:`transformers.AutoModel.from_pretrained`
        :param batch_size: >0
            the batch size to use while encoding.
        :param max_length: >0
            the maximum number of tokens to pad/trim the labels to

        :raise ImportError:
            if the transformers library could not be imported
        """
        self.tensor = TransformerEncoder(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            max_length=max_length,
        ).encode_all(
            labels=labels,
            batch_size=batch_size,
        )

    @classmethod
    def from_triples_factory(
        cls,
        triples_factory: TriplesFactory,
        for_entities: bool = True,
        **kwargs,
    ) -> "LabelBasedInitializer":
        """
        Prepare a label-based initializer with labels from a triples factory.

        :param triples_factory:
            the triples factory
        :param for_entities:
            whether to create the initializer for entities (or relations)
        :param kwargs:
            additional keyword-based arguments passed to :func:`LabelBasedInitializer.__init__`

        :raise ImportError:
            if the transformers library could not be imported
        """
        id_to_label = triples_factory.entity_id_to_label if for_entities else triples_factory.relation_id_to_label
        labels = [id_to_label[i] for i in sorted(id_to_label.keys())]
        return cls(
            labels=labels,
            **kwargs,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize the tensor from the encoded labels."""
        if x.shape != self.tensor.shape:
            raise ValueError(f"shape does not match: expected {self.tensor.shape} but got {x.shape}")
        return self.tensor.to(device=x.device, dtype=x.dtype)
