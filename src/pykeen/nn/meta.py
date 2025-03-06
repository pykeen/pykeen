"""A collection of high-level wrappers around :class:`pykeen.nn` objects."""

from __future__ import annotations

from collections.abc import Sequence

from class_resolver import HintOrType, OptionalKwargs

from .init import PretrainedInitializer
from .perceptron import TwoLayerMLP
from .representation import CombinedRepresentation, Embedding, Representation, TransformedRepresentation
from ..typing import FloatTensor

__all__ = [
    "MLPTransformedRepresentation",
    "FeatureEnrichedEmbedding",
]


class MLPTransformedRepresentation(TransformedRepresentation):
    """A representation that transforms a representation with a learnable two-layer MLP.

    In the following example, we show how to construct a feature-enriched embedding.

    .. literalinclude:: ../examples/nn/representation/mlp_transformation.py
    """

    def __init__(
        self,
        *,
        base: HintOrType[Representation] = None,
        base_kwargs: OptionalKwargs = None,
        output_dim: int | None = None,
        mlp_dropout: float = 0.1,
        ratio: int | float = 2,
        **kwargs,
    ) -> None:
        """Initialize the representation.

        :param base: the base representation, or a hint thereof, cf. `representation_resolver`
        :param base_kwargs: keyword-based parameters used to instantiate the base representation
        :param output_dim: the output dimension. defaults to input dim
        :param mlp_dropout: the dropout value on the hidden layer.

            .. warning::

                don't confuse with the optional keyword argument for the representation's dropout

        :param ratio: the ratio of the output dimension to the hidden layer size.
        :param kwargs: keyword arguments forwarded to the parent's constructor
        """
        # import here to avoid cyclic import
        from . import representation_resolver

        base = representation_resolver.make(base, base_kwargs)

        super().__init__(
            base=base,
            max_id=base.max_id,
            transformation=TwoLayerMLP(base.shape[0], output_dim=output_dim, dropout=mlp_dropout, ratio=ratio),
            **kwargs,
        )


class FeatureEnrichedEmbedding(CombinedRepresentation):
    """A combination of a static feature and a learnable representation.

    In the following example, we show how to construct a feature-enriched embedding.

    .. literalinclude:: ../examples/nn/representation/feature_enriched_embedding.py
    """

    def __init__(
        self, tensor: FloatTensor | PretrainedInitializer, shape: None | int | Sequence[int] = None, **kwargs
    ) -> None:
        """Initialize the feature-enriched embedding.

        :param tensor: the tensor of pretrained embeddings, or a pretrained initializer that wraps a tensor of
            pretrained embeddings.
        :param shape: an explicit shape for the learned embedding. If None, it is inferred from the provided feature
            tensor.
        :param kwargs: Keyword arguments passed to :meth:`pykeen.nn.CombinedRepresentation.__init__`.

            For example, if you want to make sure that the dimensions of the output are the same as the input, set
            ``combination="ConcatProjection"``. to use :class:`pykeen.nn.ConcatProjectionCombination`.
        """
        static_embedding = Embedding.from_pretrained(tensor, trainable=False)
        if shape is None:
            shape = static_embedding.shape
        trainable_embedding = Embedding(max_id=static_embedding.max_id, shape=shape)
        super().__init__(
            max_id=static_embedding.max_id,
            base=[static_embedding, trainable_embedding],
            **kwargs,
        )
