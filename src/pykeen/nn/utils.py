# -*- coding: utf-8 -*-

"""Utilities for neural network components."""

from typing import Optional, Sequence, Union

import torch
from torch import nn

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
        :param max_length: >0
            the maximum number of tokens to pad/trim the labels to
        """
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError("Please install the `transformers` library") from error

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)

        self.max_length = max_length

    @staticmethod
    def forward(self, labels: Union[str, Sequence[str]]) -> torch.FloatTensor:
        """Encode labels via the provided model & tokenizer."""
        if isinstance(labels, str):
            labels = [labels]
        max_length = self.max_length or max(map(len, labels))
        return self.model(
            **self.tokenizer(
                labels,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
        ).pooler_output
