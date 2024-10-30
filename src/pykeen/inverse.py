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
    """An interface for inverse-relation ID mapping."""

    # TODO: method is_inverse?

    @abstractmethod
    def get_inverse_id(self, relation_id: RelationID) -> RelationID:
        """Get the inverse ID for a given relation."""
        # TODO: inverse of inverse?

    @abstractmethod
    def _map(self, batch: LongTensor, index: int = 1) -> LongTensor:
        """Map relations in a batch."""

    @abstractmethod
    def invert_(self, batch: LongTensor, index: int = 1) -> LongTensor:
        """Invert relations in a batch (in-place)."""

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
    def get_inverse_id(self, relation_id: RelationID) -> RelationID:  # noqa: D102
        return relation_id + 1

    # docstr-coverage: inherited
    def _map(self, batch: LongTensor, index: int = 1) -> LongTensor:  # noqa: D102
        batch = batch.clone()
        batch[:, index] *= 2
        return batch

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
