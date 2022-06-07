"""Modules for text encoding."""


import logging
import string
from abc import abstractmethod
from typing import Callable, Optional, Sequence, Union

import torch
from class_resolver import Hint, HintOrType, ClassResolver, OptionalKwargs
from class_resolver.contrib.torch import aggregation_resolver
from more_itertools import chunked
from torch import nn
from torch_max_mem import MemoryUtilizationMaximizer
from tqdm.auto import tqdm

from .representation import Representation
from ..utils import get_preferred_device, resolve_device, upgrade_to_sequence


__all__ = [
    "TextRepresentation",
    "TextEncoder",
    "TransformerEncoder",
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

        :param labels: length: b
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


class CharacterEmbeddingEncoder(TextEncoder):
    """A simple character-based text encoder."""

    def __init__(
        self,
        dim: int = 32,
        character_representation: HintOrType[Representation] = None,
        vocabulary: str = string.printable,
        aggregation: Hint[Callable[..., torch.FloatTensor]] = None,
    ) -> None:
        """Initialize the encoder.

        :param dim: the embedding dimension
        :param character_representation: the character representation or a hint thereof
        :param vocabulary: the vocubarly, i.e., the allowed characters
        :param aggregation: the aggregation to use to pool the character embeddings
        """
        super().__init__()
        from . import representation_resolver

        self.aggregation = aggregation_resolver.make(aggregation, dim=-2)
        self.vocabulary = vocabulary
        self.token_to_id = {c: i for i, c in enumerate(vocabulary)}
        self.character_embedding = representation_resolver.make(
            character_representation, max_id=len(self.vocabulary) + 1, shape=dim
        )

    # docstr-coverage: inherited
    def forward_normalized(self, texts: Sequence[str]) -> torch.FloatTensor:  # noqa: D102
        # tokenize
        token_ids = [[self.token_to_id.get(c, -1) for c in text] for text in texts]
        # pad
        max_length = max(map(len, token_ids))
        indices = torch.full(size=(len(texts), max_length), fill_value=-1)
        for i, ids in enumerate(token_ids):
            indices[i, : len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        # get character embeddings
        x = self.character_embedding(indices=indices)
        # pool
        x = self.aggregation(x, dim=-2)
        if not torch.is_tensor(x):
            x = x.values
        return x


class TransformerEncoder(TextEncoder):
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
    default=TransformerEncoder,
)


class TextRepresentation(Representation):
    """
    Textual representations using a text encoder on labels.

    Example Usage:

    Entity representations are obtained by encoding the labels with a Transformer model. The transformer
    model becomes part of the KGE model, and its parameters are trained jointly.

    .. code-block:: python

        from pykeen.datasets import get_dataset
        from pykeen.nn.representation import EmbeddingSpecification, LabelBasedTransformerRepresentation
        from pykeen.models import ERModel

        dataset = get_dataset(dataset="nations")
        entity_representations = TextRepresentation.from_triples_factory(
            triples_factory=dataset.training,
        )
        model = ERModel(
            interaction="ermlp",
            entity_representations=entity_representations,
            relation_representations=EmbeddingSpecification(shape=entity_representations.shape),
        )
    """

    def __init__(
        self,
        labels: Sequence[str],
        encoder: HintOrType[TextEncoder] = None,
        encoder_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        """
        Initialize the representation.

        :param labels:
            the labels
        :param encoder:
            the text encoder, or a hint thereof
        :param encoder_kwargs:
            keyword-based parameters used to instantiate the text encoder
        :param kwargs:
            additional keyword-based parameters passed to super.__init__
        """
        text_encoder_resolver.make(encoder, encoder_kwargs)
        # infer shape
        shape = encoder.encode_all(labels[0:1]).shape[1:]
        super().__init__(max_id=len(labels), shape=shape, **kwargs)
        self.labels = labels
        # assign after super, since they should be properly registered as submodules
        self.encoder = encoder

    @classmethod
    def from_triples_factory(
        cls,
        triples_factory: TriplesFactory,
        for_entities: bool = True,
        **kwargs,
    ) -> "TextRepresentation":
        """
        Prepare a label-based transformer representations with labels from a triples factory.

        :param triples_factory:
            the triples factory
        :param for_entities:
            whether to create the initializer for entities (or relations)
        :param kwargs:
            additional keyword-based arguments passed to :func:`LabelBasedTransformerRepresentation.__init__`

        :returns:
            A label-based transformer from the triples factory

        :raise ImportError:
            if the transformers library could not be imported
        """
        id_to_label = triples_factory.entity_id_to_label if for_entities else triples_factory.relation_id_to_label
        return cls(
            labels=[id_to_label[i] for i in range(len(id_to_label))],
            **kwargs,
        )

    # docstr-coverage: inherited
    def _plain_forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        if indices is None:
            indices = torch.arange(self.max_id, device=self.device)
        uniq, inverse = indices.to(device=self.device).unique(return_inverse=True)
        x = self.encoder(
            labels=[self.labels[i] for i in uniq.tolist()],
        )
        return x[inverse]
