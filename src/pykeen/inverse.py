"""Relation inversion logic."""

from abc import ABC, abstractmethod
from typing import TypeVar, overload

from class_resolver import Resolver

from .typing import BoolTensor, LongTensor

__all__ = [
    "RelationID",
    "RelationInverter",
    "DefaultRelationInverter",
    "relation_inverter_resolver",
]

#: A relation ID, either a single one, or a batch thereof
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
        """Get the internal ID of the inverse of an internal relation ID.

        Implementations have to be involutions, i.e., applying this method twice has to yield the
        input ID again. In particular, it maps the ID of an inverse relation back to its forward one.
        """

    @overload
    def is_inverse(self, relation_id: int) -> bool: ...

    @overload
    def is_inverse(self, relation_id: LongTensor) -> BoolTensor: ...

    @abstractmethod
    def is_inverse(self, relation_id: RelationID) -> bool | BoolTensor:
        """Return whether the internal relation IDs correspond to inverse relations."""

    def invert_internal_batch(self, batch: LongTensor, index: int = 1) -> LongTensor:
        """Invert the internal relation IDs in a batch, leaving the input batch unmodified."""
        batch = batch.clone()
        batch[:, index] = self.get_inverse_id(batch[:, index])
        return batch

    def to_internal_batch(self, batch: LongTensor, index: int = 1, invert: bool = False) -> LongTensor:
        """Convert the real relation IDs in a batch to internal ones, optionally also inverting them.

        The input batch is left unmodified.
        """
        relation_id = self.to_internal(batch[:, index])
        if invert:
            relation_id = self.get_inverse_id(relation_id)
        batch = batch.clone()
        batch[:, index] = relation_id
        return batch


class DefaultRelationInverter(RelationInverter):
    """Uses the lowest bit of the internal relation ID as the "is inverse" flag.

    The internal ID is the real relation ID shifted up by one bit, with the lowest bit set for
    inverse relations, i.e., ``internal_id = (real_id << 1) | is_inverse``. Equivalently, forward
    relations get even IDs, and the corresponding inverse the next odd one.
    """

    # docstr-coverage: inherited
    def to_internal(self, relation_id: RelationID) -> RelationID:  # noqa: D102
        # shift up to make room for the flag, which is unset for forward relations
        return relation_id << 1

    # docstr-coverage: inherited
    def to_real(self, relation_id: RelationID) -> RelationID:  # noqa: D102
        # shift the flag out again
        return relation_id >> 1

    # docstr-coverage: inherited
    def get_inverse_id(self, relation_id: RelationID) -> RelationID:  # noqa: D102
        # toggling the flag switches between a relation and its inverse in either direction
        return relation_id ^ 1

    # docstr-coverage: inherited
    def is_inverse(self, relation_id: RelationID) -> bool | BoolTensor:  # noqa: D102
        return (relation_id & 1) == 1


#: A resolver for relation inverter protocols
relation_inverter_resolver: Resolver[RelationInverter] = Resolver.from_subclasses(
    RelationInverter,
    default=DefaultRelationInverter,
)
