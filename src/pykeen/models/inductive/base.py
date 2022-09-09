"""Base classes for inductive models."""
from typing import Optional, Sequence

from class_resolver import OneOrManyHintOrType, OneOrManyOptionalKwargs

from ..nbase import ERModel
from ...nn import Representation
from ...triples import CoreTriplesFactory
from ...typing import TESTING, TRAINING, VALIDATION, InductiveMode


class InductiveERModel(ERModel):
    """A base class for inductive models."""

    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        entity_representations: OneOrManyHintOrType[Representation] = None,
        entity_representations_kwargs: OneOrManyOptionalKwargs = None,
        # inductive factories
        validation_factory: CoreTriplesFactory,
        testing_factory: Optional[CoreTriplesFactory] = None,
        **kwargs,
    ) -> None:
        """Initialize the inductive model.

        :param triples_factory:
            the (training) factory
        :param entity_representations:
            the training entity representations
        :param entity_representations_kwargs:
            additional keyword-based parameters for the training entity representations
        :param validation_factory:
            the validation factory
        :param testing_factory:
            the testing factory. If None, the validation factory is re-used, i.e., validation and test entities come
            from the same (unseen) set of entities.
        :param kwargs:
            additional keyword-based parameters passed to :meth:`ERModel.__init__`
        """
        super().__init__(
            triples_factory=triples_factory,
            entity_representations=entity_representations,
            entity_representations_kwargs=entity_representations_kwargs,
            **kwargs,
        )
        # note: this is *not* a nn.ModuleDict; the modules have to be registered elsewhere
        self._mode_to_representations = {TRAINING: self.entity_representations}
        self._mode_to_representations[VALIDATION] = validation_entity_representations = self._build_representations(
            triples_factory=validation_factory,
            entity_representations=entity_representations,
            entity_representations_kwargs=entity_representations_kwargs,
        )

        # shared
        if testing_factory is None:
            testing_entity_representations = validation_entity_representations
        else:
            # non-shared
            testing_entity_representations = self._build_representations(
                triples_factory=testing_factory,
                entity_representations=entity_representations,
                entity_representations_kwargs=entity_representations_kwargs,
            )
        self._mode_to_representations[TESTING] = testing_entity_representations

    # docstr-coverage: inherited
    def _get_entity_representations_from_inductive_mode(
        self, *, mode: Optional[InductiveMode]
    ) -> Sequence[Representation]:  # noqa: D102
        if mode in self._mode_to_representations:
            return self._mode_to_representations[mode]
        elif mode is None:
            raise ValueError(f"{self.__class__.__name__} does not support inductive mode: {mode}")
        else:
            raise ValueError(f"Invalid mode: {mode}")

    # docstr-coverage: inherited
    def _get_entity_len(self, *, mode: Optional[InductiveMode]) -> Optional[int]:  # noqa: D102
        return self._get_entity_representations_from_inductive_mode(mode=mode)[0].max_id
