"""Relation inversion logic."""

from abc import ABC, abstractmethod
from typing import TypeVar

from class_resolver import Resolver

from .typing import BoolTensor, LongTensor

__all__ = [
    "RelationInverter",
    "DefaultRelationInverter",
    "relation_inverter_resolver",
]

RelationID = TypeVar("RelationID", int, LongTensor)


class RelationInverter(ABC):
    """An interface for inverse-relation ID mapping.

    Implementations translate between two ID spaces:

    - the *real* relation IDs, ``0 ... num_real_relations - 1``, as used by the triples stored
      inside a :class:`~pykeen.triples.CoreTriplesFactory`, and
    - the *internal* relation IDs, ``0 ... 2 * num_real_relations - 1``, as used by models trained
      with inverse relations, which comprise a forward and an inverse ID for each real relation.

    Subclasses only define the ID-level translation; the batch-level operations are derived from it.
    """

    @abstractmethod
    def to_internal(self, relation_id: RelationID) -> RelationID:
        """Convert a real relation ID to the corresponding internal (forward) relation ID."""

    @abstractmethod
    def to_real(self, relation_id: RelationID) -> RelationID:
        """Convert an internal relation ID to the corresponding real relation ID."""

    @abstractmethod
    def get_inverse_id(self, relation_id: RelationID) -> RelationID:
        """Get the internal inverse ID for a given internal (forward) relation ID."""
        # TODO: inverse of inverse?

    @abstractmethod
    def is_inverse(self, ids: LongTensor) -> BoolTensor:
        """Return a mask whether the relation IDs correspond to inverse relations."""

    def invert(self, batch: LongTensor, index: int = 1) -> LongTensor:
        """Invert the (internal) relations in a batch, leaving the input batch unmodified."""
        batch = batch.clone()
        batch[:, index] = self.get_inverse_id(batch[:, index])
        return batch

    def map(self, batch: LongTensor, index: int = 1, invert: bool = False) -> LongTensor:
        """Map relations in a batch from real to internal IDs, optionally also inverting them.

        The input batch is left unmodified.
        """
        relation_id = self.to_internal(batch[:, index])
        if invert:
            relation_id = self.get_inverse_id(relation_id)
        batch = batch.clone()
        batch[:, index] = relation_id
        return batch


class DefaultRelationInverter(RelationInverter):
    """Maps normal relations to even IDs, and the corresponding inverse to the next odd ID."""

    # docstr-coverage: inherited
    def to_internal(self, relation_id: RelationID) -> RelationID:  # noqa: D102
        return 2 * relation_id

    # docstr-coverage: inherited
    def to_real(self, relation_id: RelationID) -> RelationID:  # noqa: D102
        return relation_id // 2

    # docstr-coverage: inherited
    def get_inverse_id(self, relation_id: RelationID) -> RelationID:  # noqa: D102
        return relation_id + 1

    # docstr-coverage: inherited
    def is_inverse(self, ids: LongTensor) -> BoolTensor:  # noqa: D102
        return ids % 2 == 1


#: A resolver for relation inverter protocols
relation_inverter_resolver: Resolver[RelationInverter] = Resolver.from_subclasses(
    RelationInverter,
    default=DefaultRelationInverter,
)
