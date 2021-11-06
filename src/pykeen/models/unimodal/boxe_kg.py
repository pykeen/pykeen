from typing import Any, ClassVar, Mapping, Optional

from torch.nn.init import uniform_

from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...models import ERModel
from ...nn.emb import EmbeddingSpecification
from ...nn.init import uniform_norm_
from ...nn.modules import BoxEKGInteraction
from ...typing import Hint, Initializer

__all__ = [
    "BoxEKG",
]


class BoxEKG(ERModel):
    r"""An implementation of BoxE.

    ---
    citation:
        author: Abboud
        year: 2020
        link: https://arxiv.org/abs/2007.06267
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        p=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        tanh_map: bool = True,
        norm_order: int = 2,
        entity_initializer: Hint[Initializer] = uniform_norm_,
        entity_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Initializer] = uniform_norm_,  # Has to be scaled as well
        relation_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_size_initializer: Hint[Initializer] = uniform_,  # Has to be scaled as well
        relation_size_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        r"""Initialize BoxE-KG

        :param embedding_dim: The entity embedding dimension $d$. Defaults to 200. Is usually $d \in [50, 300]$.
        :param tanh_map: Whether to use tanh mapping after BoxE computation.
        tanh mapping restricts the embedding space to the range [-1, 1], and thus this map implicitly
        regularizes the space to prevent loss reduction by growing boxes arbitrarily large. Default - True
        :param norm_order: Norm Order in score computation (Int): Default - 2
        :param entity_initializer: Entity initializer function. Defaults to :func:`pykeen.nn.init.uniform_norm_`
        :param entity_initializer_kwargs: Keyword arguments to be used when calling the entity initializer
        :param relation_initializer: Relation initializer function. Defaults to :func:`pykeen.nn.init.uniform_norm_`
        :param relation_initializer_kwargs: Keyword arguments to be used when calling the relation initializer
        :param relation_size_initializer: Relation initializer function. Defaults to :func:`torch.nn.init.uniform_`
            Defaults to :func:`torch.nn.init.uniform_`
        :param relation_size_initializer_kwargs: Keyword arguments to be used when calling the
            relation matrix initializer
        """

        super().__init__(
            interaction=BoxEKGInteraction,
            interaction_kwargs=dict(norm_order=norm_order, tanh_map=tanh_map),
            entity_representations=[  # Base position
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=entity_initializer,
                    initializer_kwargs=entity_initializer_kwargs,
                ),  # Bump
                # entity bias for head
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=entity_initializer,
                    initializer_kwargs=entity_initializer_kwargs,
                ),
            ],
            relation_representations=[
                # relation position head
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs,
                ),
                # relation shape head
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs,
                ),
                EmbeddingSpecification(
                    embedding_dim=1,  # Size
                    initializer=relation_size_initializer,
                    initializer_kwargs=relation_size_initializer_kwargs,
                ),
                EmbeddingSpecification(  # Tail position
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs,
                ),
                # relation shape tail
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs,
                ),
                EmbeddingSpecification(
                    embedding_dim=1,  # Tail Size
                    initializer=relation_size_initializer,
                    initializer_kwargs=relation_size_initializer_kwargs,
                ),
            ],
            **kwargs,
        )
