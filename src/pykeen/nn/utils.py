# -*- coding: utf-8 -*-

"""Utilities for neural network components."""

from typing import Optional, Sequence, Union

import torch
from more_itertools import chunked
from torch import nn
from tqdm.auto import tqdm

from ..utils import get_preferred_device

__all__ = [
    "TransformerEncoder",
]


class TransformerEncoder(nn.Module):
    """A combination of a tokenizer and a model."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_length: Optional[int] = None,
    ):
        """
        Initialize the encoder.

        :param pretrained_model_name_or_path:
            the name of the pretrained model, or a path, cf. :func:`transformers.AutoModel.from_pretrained`
        :param max_length: >0, default: 512
            the maximum number of tokens to pad/trim the labels to

        :raises ImportError:
            if the :mod:`transformers` library could not be imported
        """
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer
        except ImportError as error:
            raise ImportError(
                "Please install the `transformers` library, use the _transformers_ extra"
                " for PyKEEN with `pip install pykeen[transformers] when installing, or "
                " see the PyKEEN installation docs at https://pykeen.readthedocs.io/en/stable/installation.html"
                " for more information."
            ) from error

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        self.max_length = max_length or 512

    def forward(self, labels: Union[str, Sequence[str]]) -> torch.FloatTensor:
        """Encode labels via the provided model and tokenizer."""
        if isinstance(labels, str):
            labels = [labels]
        return self.model(
            **self.tokenizer(
                labels,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(get_preferred_device(self.model))
        ).pooler_output

    @torch.inference_mode()
    def encode_all(
        self,
        labels: Sequence[str],
        batch_size: int = 1,
    ) -> torch.FloatTensor:
        """Encode all labels (inference mode & batched).

        :param labels:
            a sequence of strings to encode
        :param batch_size:
            the batch size to use for encoding the labels. ``batch_size=1``
            means that the labels are encoded one-by-one, while ``batch_size=len(labels)``
            would correspond to encoding all at once.
            Larger batch sizes increase memory requirements, but may be computationally
            more efficient.

        :returns: shape: (len(labels), dim)
            a tensor representing the encodings for all labels
        """
        return torch.cat(
            [self(batch) for batch in chunked(tqdm(labels), batch_size)],
            dim=0,
        )
