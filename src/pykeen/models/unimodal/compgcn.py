"""Implementation of the Comp-GCN model."""

from collections.abc import Mapping
from typing import Any, ClassVar

from class_resolver import Hint, HintOrType

from ..nbase import CUSTOMIZING_REPRESENTATIONS_HINT, ERModel
from ...inverse import RelationInverter, relation_inverter_resolver
from ...nn.modules import DistMultInteraction, Interaction
from ...nn.representation import CombinedCompGCNRepresentations
from ...triples import CoreTriplesFactory
from ...typing import FloatTensor, RelationRepresentation
from ...utils import forbid_kwargs

__all__ = [
    "CompGCN",
]


class CompGCN(ERModel[FloatTensor, RelationRepresentation, FloatTensor]):
    """An implementation of CompGCN from [vashishth2020]_.

    This model uses graph convolutions, and composition functions.

    ---
    citation:
        author: Vashishth
        year: 2020
        link: https://arxiv.org/pdf/1911.03082
        github: malllabiisc/CompGCN
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = {
        "embedding_dim": {"type": int, "low": 32, "high": 512, "q": 32},
    }

    @forbid_kwargs("entity_representations", "relation_representations", message=CUSTOMIZING_REPRESENTATIONS_HINT)
    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        embedding_dim: int = 64,
        encoder_kwargs: Mapping[str, Any] | None = None,
        interaction: Hint[Interaction[FloatTensor, RelationRepresentation, FloatTensor]] = None,
        interaction_kwargs: Mapping[str, Any] | None = None,
        use_inverse_triples: bool = True,
        relation_inverter: HintOrType[RelationInverter] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the model.

        :param triples_factory:
            The triples factory.
        :param embedding_dim:
            The embedding dimension to be used if ``embedding_specification`` is not given explicitly in
            ``encoder_kwargs``.
        :param encoder_kwargs:
            Additional keyword arguments for the encoder,
            cf. :class:`~pykeen.nn.representation.CombinedCompGCNRepresentations`.
        :param interaction:
            The interaction function to use as decoder.
        :param interaction_kwargs:
            Additional keyword based arguments for the interaction function.
        :param use_inverse_triples:
            Whether to use inverse relations. Must be True, since the CompGCN encoder always creates representations
            for inverse relations.
        :param kwargs:
            Additional keyword based arguments passed to :class:`~pykeen.models.ERModel`.

        :raises ValueError:
            if ``use_inverse_triples`` is False
        """
        if not use_inverse_triples:
            raise ValueError("CompGCN requires inverse relations. Create the model with use_inverse_triples=True.")

        encoder_kwargs = {} if encoder_kwargs is None else dict(encoder_kwargs)
        encoder_kwargs.setdefault("entity_representations_kwargs", {"embedding_dim": embedding_dim})
        encoder_kwargs.setdefault("relation_representations_kwargs", encoder_kwargs["entity_representations_kwargs"])

        # the encoder has to use the same inverse relation ID scheme as the model
        if "relation_inverter" in encoder_kwargs:
            raise ValueError(
                "Do not pass a relation inverter via encoder_kwargs; the encoder uses the model's relation "
                "inverter. Pass it as relation_inverter=... to CompGCN instead."
            )
        relation_inverter = relation_inverter_resolver.make(relation_inverter)

        # combined representation
        combined = CombinedCompGCNRepresentations(
            triples_factory=triples_factory,
            relation_inverter=relation_inverter,
            **encoder_kwargs,
        )
        entity_representations, relation_representations = combined.split()

        # Resolve interaction function
        if interaction is None:
            interaction = DistMultInteraction
        super().__init__(
            triples_factory=triples_factory,
            use_inverse_triples=use_inverse_triples,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
            entity_representations=entity_representations,
            relation_representations=relation_representations,
            relation_inverter=relation_inverter,
            **kwargs,
        )
