"""Modules for text encoding."""


import logging
import string
from abc import abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Union

import torch
from class_resolver import ClassResolver, Hint, HintOrType
from class_resolver.contrib.torch import aggregation_resolver
from more_itertools import chunked
from torch import nn
from torch_max_mem import MemoryUtilizationMaximizer
from tqdm.auto import tqdm

from ..utils import get_preferred_device, resolve_device, upgrade_to_sequence

if TYPE_CHECKING:
    from .representation import Representation

__all__ = [
    # abstract
    "TextEncoder",
    "text_encoder_resolver",
    # concrete
    "CharacterEmbeddingTextEncoder",
    "TransformerTextEncoder",
]

logger = logging.getLogger(__name__)
memory_utilization_maximizer = MemoryUtilizationMaximizer()


@memory_utilization_maximizer
def _encode_all_memory_utilization_optimized(
    encoder: "TextEncoder",
    labels: Sequence[str],
    batch_size: int,
) -> torch.Tensor:
    """
    Encode all labels with the given batch-size.

    Wrapped by memory utilization maximizer to automatically reduce the batch size if needed.

    :param encoder:
        the encoder
    :param labels:
        the labels to encode
    :param batch_size:
        the batch size to use. Will automatically be reduced if necessary.

    :return: shape: `(len(labels), dim)`
        the encoded labels
    """
    return torch.cat(
        [encoder(batch) for batch in chunked(tqdm(map(str, labels), leave=False), batch_size)],
        dim=0,
    )


class TextEncoder(nn.Module):
    """An encoder for text."""

    def forward(self, labels: Union[str, Sequence[str]]) -> torch.FloatTensor:
        """
        Encode a batch of text.

        :param labels: length: b
            the texts

        :return: shape: `(b, dim)`
            an encoding of the texts
        """
        labels = upgrade_to_sequence(labels)
        labels = list(map(str, labels))
        return self.forward_normalized(texts=labels)

    @abstractmethod
    def forward_normalized(self, texts: Sequence[str]) -> torch.FloatTensor:
        """
        Encode a batch of text.

        :param texts: length: b
            the texts

        :return: shape: `(b, dim)`
            an encoding of the texts
        """
        raise NotImplementedError

    @torch.inference_mode()
    def encode_all(
        self,
        labels: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        """Encode all labels (inference mode & batched).

        :param labels:
            a sequence of strings to encode
        :param batch_size:
            the batch size to use for encoding the labels. ``batch_size=1``
            means that the labels are encoded one-by-one, while ``batch_size=len(labels)``
            would correspond to encoding all at once.
            Larger batch sizes increase memory requirements, but may be computationally
            more efficient. `batch_size` can also be set to `None` to enable automatic batch
            size maximization for the employed hardware.

        :returns: shape: (len(labels), dim)
            a tensor representing the encodings for all labels
        """
        return _encode_all_memory_utilization_optimized(
            encoder=self, labels=labels, batch_size=batch_size or len(labels)
        ).detach()


class CharacterEmbeddingTextEncoder(TextEncoder):
    """
    A simple character-based text encoder.

    This encoder uses base representations for each character from a given alphabet, as well as two special tokens
    for unknown character and padding. To encoder a sentence, it converts it to a sequence of characters, obtains
    the invidual characters representations and aggregates these representations to a single one.

    With :class:`pykeen.nn.representation.Embedding` character representation and :func:`torch.mean` aggregation,
    this encoder is similar to a bag-of-characters model with trainable character embeddings. Therefore, it is
    invariant to the ordering of characters:

    >>> from pykeen.nn.text import CharacterEmbeddingTextEncoder
    >>> encoder = CharacterEmbeddingTextEncoder()
    >>> import torch
    >>> torch.allclose(encoder("seal"), encoder("sale"))
    True
    """

    def __init__(
        self,
        dim: int = 32,
        character_representation: HintOrType["Representation"] = None,
        vocabulary: str = string.printable,
        aggregation: Hint[Callable[..., torch.FloatTensor]] = None,
    ) -> None:
        """Initialize the encoder.

        :param dim: the embedding dimension
        :param character_representation: the character representation or a hint thereof
        :param vocabulary: the vocabulary, i.e., the allowed characters
        :param aggregation: the aggregation to use to pool the character embeddings
        """
        super().__init__()
        from . import representation_resolver

        self.aggregation = aggregation_resolver.make(aggregation, dim=-2)
        self.vocabulary = vocabulary
        self.token_to_id = {c: i for i, c in enumerate(vocabulary)}
        num_real_tokens = len(self.vocabulary)
        self.unknown_idx = num_real_tokens
        self.padding_idx = num_real_tokens + 1
        self.character_embedding = representation_resolver.make(
            character_representation, max_id=num_real_tokens + 2, shape=dim
        )

    # docstr-coverage: inherited
    def forward_normalized(self, texts: Sequence[str]) -> torch.FloatTensor:  # noqa: D102
        # tokenize
        token_ids = [[self.token_to_id.get(c, self.unknown_idx) for c in text] for text in texts]
        # pad
        max_length = max(map(len, token_ids))
        indices = torch.full(size=(len(texts), max_length), fill_value=self.padding_idx)
        for i, ids in enumerate(token_ids):
            indices[i, : len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        # get character embeddings
        x = self.character_embedding(indices=indices)
        # pool
        x = self.aggregation(x)
        if not torch.is_tensor(x):
            x = x.values
        return x


class TransformerTextEncoder(TextEncoder):
    """A combination of a tokenizer and a model."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = "bert-base-cased",
        max_length: int = 512,
    ):
        """
        Initialize the encoder using :class:`transformers.AutoModel`.

        :param pretrained_model_name_or_path:
            the name of the pretrained model, or a path, cf. :meth:`transformers.AutoModel.from_pretrained`
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
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path).to(
            resolve_device()
        )
        self.max_length = max_length or 512

    # docstr-coverage: inherited
    def forward_normalized(self, texts: Sequence[str]) -> torch.FloatTensor:  # noqa: D102
        return self.model(
            **self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(get_preferred_device(self.model))
        ).pooler_output


text_encoder_resolver: ClassResolver[TextEncoder] = ClassResolver.from_subclasses(
    base=TextEncoder,
    default=CharacterEmbeddingTextEncoder,
)
