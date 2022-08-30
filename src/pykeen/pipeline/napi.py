"""New-Style Pipeline API."""
import dataclasses
from typing import Any, Dict, Literal, Optional

from class_resolver import HintOrType, OneOrManyHintOrType, OneOrManyOptionalKwargs, OptionalKwargs

from pykeen.losses import Loss, loss_resolver
from pykeen.models import ERModel
from pykeen.models.base import Model
from pykeen.models.nbase import _prepare_representation_module_list
from pykeen.nn.modules import Interaction, interaction_resolver
from pykeen.nn.representation import Representation
from pykeen.triples.triples_factory import KGInfo


def _default_representation_kwargs():
    """Return default representation kwargs."""
    return dict(shape=(32,))


@dataclasses.dataclass
class ModelBuilder:
    """A state-ful model builder."""

    # KG info
    triples_factory: Optional[KGInfo] = KGInfo(num_entities=7, num_relations=2, create_inverse_triples=False)

    # interaction
    interaction: HintOrType[Interaction] = None
    interaction_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # representations: entities
    entity_representations: OneOrManyHintOrType[Representation] = None
    entity_representations_kwargs: OneOrManyOptionalKwargs = dataclasses.field(
        default_factory=_default_representation_kwargs
    )

    # representations: relations
    relation_representations = None
    relation_representations_kwargs: OneOrManyOptionalKwargs = dataclasses.field(
        default_factory=_default_representation_kwargs
    )

    # loss
    loss: HintOrType[Loss] = None
    loss_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # other
    predict_with_sigmoid: bool = False
    random_seed: Optional[int] = None

    def _resolve_interaction(self):
        self.interaction = interaction_resolver.make(self.interaction, pos_kwargs=self.interaction_kwargs)
        self.interaction_kwargs = None

    def _resolve_entity_representations(self):
        self.entity_representations = _prepare_representation_module_list(
            max_id=self.triples_factory.num_entities,
            shapes=self.interaction.full_entity_shapes(),
            label="entity",
            representations=self.entity_representations,
            representation_kwargs=self.entity_representations_kwargs,
        )
        self.entity_representations_kwargs = None

    def _resolve_relation_representations(self):
        self.relation_representations = _prepare_representation_module_list(
            max_id=self.triples_factory.num_relations,
            shapes=self.interaction.relation_shape,
            label="relation",
            representations=self.relation_representations,
            representation_kwargs=self.relation_representations_kwargs,
        )
        self.relation_representations_kwargs = None

    def _resolve_loss(self):
        self.loss = loss_resolver.make(self.loss, pos_kwargs=self.loss_kwargs)
        self.loss_kwargs = None

    def resolve(self, component: Literal[None, "interaction", "entities", "relations", "loss"] = None):
        """Resolve all components."""
        # resolve interaction first, to enable shape verification
        if component is None or component == "interaction":
            self._resolve_interaction()
        if component is None or component == "entities":
            self._resolve_entity_representations()
        if component is None or component == "relations":
            self._resolve_relation_representations()
        if component is None or component == "loss":
            self._resolve_loss()

    def build(self) -> Model:
        return ERModel(**dataclasses.asdict(self))
