# -*- coding: utf-8 -*-

"""Embedding weight initialization routines."""

import functools
import logging
import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn
import torch.nn.init
from class_resolver import FunctionResolver
from torch.nn import functional

from .utils import TransformerEncoder
from ..triples import TriplesFactory
from ..utils import compose

__all__ = [
    "xavier_uniform_",
    "xavier_uniform_norm_",
    "xavier_normal_",
    "xavier_normal_norm_",
    "uniform_norm_",
    "uniform_norm_p1_",
    "normal_norm_",
    "init_phases",
    "PretrainedInitializer",
    "LabelBasedInitializer",
    "initializer_resolver",
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
    name="xavier_uniform_norm_",
)
xavier_normal_norm_ = compose(
    torch.nn.init.xavier_normal_,
    functional.normalize,
    name="xavier_normal_norm_",
)
uniform_norm_ = compose(
    torch.nn.init.uniform_,
    functional.normalize,
    name="uniform_norm_",
)
normal_norm_ = compose(
    torch.nn.init.normal_,
    functional.normalize,
    name="normal_norm_",
)
uniform_norm_p1_ = compose(
    torch.nn.init.uniform_,
    functools.partial(functional.normalize, p=1),
    name="uniform_norm_p1_",
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


class PretrainedInitializer:
    """
    Initialize tensor with pretrained weights.

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
                entity_initializer=PretrainedInitializer(tensor=pretrained_embedding_tensor),
            ),
        )
    """

    def __init__(self, tensor: torch.FloatTensor) -> None:
        """
        Initialize the initializer.

        :param tensor:
            the tensor of pretrained embeddings.
        """
        self.tensor = tensor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize the tensor with the given tensor."""
        if x.shape != self.tensor.shape:
            raise ValueError(f"shape does not match: expected {self.tensor.shape} but got {x.shape}")
        return self.tensor.to(device=x.device, dtype=x.dtype)


class LabelBasedInitializer(PretrainedInitializer):
    """
    An initializer using pretrained models from the `transformers` library to encode labels.

    Example Usage:

    Initialize entity representations as Transformer encodings of their labels. Afterwards,
    the parameters are detached from the labels, and trained on the KGE task without any
    further connection to the Transformer model.

    .. code-block :: python

        from pykeen.datasets import get_dataset
        from pykeen.nn.init import LabelBasedInitializer
        from pykeen.models import ERMLPE

        dataset = get_dataset(dataset="nations")
        model = ERMLPE(
            embedding_dim=768,  # for BERT base
            entity_initializer=LabelBasedInitializer.from_triples_factory(
                triples_factory=dataset.training,
            ),
        )
    """

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
        super().__init__(
            tensor=TransformerEncoder(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                max_length=max_length,
            ).encode_all(
                labels=labels,
                batch_size=batch_size,
            ),
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
        :returns:
            A label-based initializer

        :raise ImportError:
            if the transformers library could not be imported
        """
        id_to_label = triples_factory.entity_id_to_label if for_entities else triples_factory.relation_id_to_label
        labels = [id_to_label[i] for i in sorted(id_to_label.keys())]
        return cls(
            labels=labels,
            **kwargs,
        )


initializer_resolver = FunctionResolver(
    [
        getattr(torch.nn.init, func)
        for func in dir(torch.nn.init)
        if not func.startswith("_") and func.endswith("_") and func not in {"xavier_normal_", "xavier_uniform_"}
    ],
    default=torch.nn.init.normal_,
)
for func in [
    xavier_normal_,
    xavier_uniform_,
    init_phases,
    xavier_normal_norm_,
    xavier_uniform_norm_,
    normal_norm_,
    uniform_norm_,
]:
    initializer_resolver.register(func)
