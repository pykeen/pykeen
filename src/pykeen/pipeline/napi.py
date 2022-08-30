"""New-Style Pipeline API."""
import dataclasses
from typing import Optional

from class_resolver import HintOrType, OneOrManyHintOrType, OneOrManyOptionalKwargs, OptionalKwargs

from pykeen.losses import Loss, loss_resolver
from pykeen.models import ERModel
from pykeen.models.base import Model
from pykeen.models.nbase import _prepare_representation_module_list
from pykeen.nn.modules import Interaction, interaction_resolver
from pykeen.nn.representation import Representation
from pykeen.triples.triples_factory import KGInfo


@dataclasses.dataclass
class ModelBuilder:
    """A state-ful model builder."""

    defer: bool = False

    # KG info
    kg_info: Optional[KGInfo] = None

    # interaction
    interaction: HintOrType[Interaction] = None
    interaction_kwargs: OptionalKwargs = None

    # representations: entities
    entity_representations: OneOrManyHintOrType[Representation] = None
    entity_representations_kwargs: OneOrManyOptionalKwargs = None

    # representations: relations
    relation_representations = None
    relation_representations_kwargs: OneOrManyOptionalKwargs = None

    # loss
    loss: HintOrType[Loss] = None
    loss_kwargs: OptionalKwargs = None

    # other
    predict_with_sigmoid: bool = False
    random_seed: Optional[int] = None

    # interaction
    def set_interaction_(self, interaction: HintOrType[Interaction], kwargs: OptionalKwargs = None):
        self.interaction, self.interaction_kwargs = interaction, kwargs
        if not self.defer:
            self._resolve_interaction()

    def _resolve_interaction(self):
        self.interaction = interaction_resolver.make(self.interaction, pos_kwargs=self.interaction_kwargs)
        self.interaction_kwargs = None

    # entities
    def set_entity_representations(self, representations: OneOrManyHintOrType, kwargs: OneOrManyOptionalKwargs = None):
        self.entity_representations, self.entity_representations_kwargs = representations, kwargs
        if not self.defer:
            self._resolve_entity_representations()

    def _resolve_entity_representations(self):
        self.entity_representations = _prepare_representation_module_list(
            max_id=self.kg_info.num_entities,
            shapes=self.interaction.full_entity_shapes(),
            label="entity",
            representations=self.entity_representations,
            representation_kwargs=self.entity_representations_kwargs,
        )
        self.entity_representations_kwargs = None

    # relations
    def set_relation_representations(
        self, representations: OneOrManyHintOrType, kwargs: OneOrManyOptionalKwargs = None
    ):
        self.relation_representations, self.relation_representations_kwargs = representations, kwargs
        if not self.defer:
            self._resolve_relation_representations()

    def _resolve_relation_representations(self):
        self.relation_representations = _prepare_representation_module_list(
            max_id=self.kg_info.num_relations,
            shapes=self.interaction.relation_shape,
            label="relation",
            representations=self.relation_representations,
            representation_kwargs=self.relation_representations_kwargs,
        )
        self.relation_representations_kwargs = None

    # loss
    def set_loss(self, loss: HintOrType[Loss], kwargs: OptionalKwargs = None):
        self.loss, self.loss_kwargs = loss, kwargs

    def _resolve_loss(self):
        self.loss = loss_resolver.make(self.loss, pos_kwargs=self.loss_kwargs)
        self.loss_kwargs = None

    def resolve(self):
        """Resolve all components."""
        # resolve interaction first, to enable shape verification
        self._resolve_interaction()
        self._resolve_entity_representations()
        self._resolve_relation_representations()
        self._resolve_loss()

    def build(self) -> Model:
        return ERModel(**dataclasses.asdict(self))
