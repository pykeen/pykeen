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

    def _map(self, batch: LongTensor, index: int = 1) -> LongTensor:
        """Map relations in a batch from real to internal IDs."""
        batch = batch.clone()
        batch[:, index] = self.to_internal(batch[:, index])
        return batch

    @abstractmethod
    def invert_(self, batch: LongTensor, index: int = 1) -> LongTensor:
        """Invert relations in a batch (in-place)."""

    def invert(self, batch: LongTensor, index: int = 1) -> LongTensor:
        """Invert relations in a batch, leaving the input batch unmodified."""
        return self.invert_(batch=batch.clone(), index=index)

    def map(self, batch: LongTensor, index: int = 1, invert: bool = False) -> LongTensor:
        """Map relations in a batch, optionally also inverting them."""
        batch = self._map(batch=batch, index=index)
        return self.invert_(batch=batch, index=index) if invert else batch

    @abstractmethod
    def is_inverse(self, ids: LongTensor) -> BoolTensor:
        """Return a mask whether the relation IDs correspond to inverse relations."""


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
    def invert_(self, batch: LongTensor, index: int = 1) -> LongTensor:  # noqa: D102
        # The number of relations stored in the triples factory includes the number of inverse relations
        # Id of inverse relation: relation + 1
        batch[:, index] += 1
        return batch

    # docstr-coverage: inherited
    def is_inverse(self, ids: LongTensor) -> BoolTensor:  # noqa: D102
        return ids % 2 == 1


#: A resolver for relation inverter protocols
relation_inverter_resolver: Resolver[RelationInverter] = Resolver.from_subclasses(
    RelationInverter,
    default=DefaultRelationInverter,
)
