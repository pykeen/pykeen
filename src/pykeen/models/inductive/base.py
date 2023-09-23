"""Base classes for inductive models."""
from collections import ChainMap
from typing import Mapping, Optional, Sequence

from class_resolver import OneOrManyHintOrType, OneOrManyOptionalKwargs
from torch import nn

from ..nbase import ERModel
from ...nn import Representation
from ...triples import CoreTriplesFactory
from ...typing import TESTING, TRAINING, VALIDATION, InductiveMode


class InductiveERModel(ERModel):
    """
    A base class for inductive models.

    This model assumes a shared set of relations between all triple sets (e.g., training and validation), and a
    separate inference factory used during validation. During testing time, either the validation factory is re-used
    or another separate testing factory may be provided.
    """

    #: a mapping from inductive mode to corresponding entity representations
    #: note: there may be duplicate values, if entity representations are shared between validation and testing
    _mode_to_representations: Mapping[str, Sequence[Representation]]

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
        # entity representation kwargs may contain a triples factory, which needs to be replaced
        entity_representations_kwargs = entity_representations_kwargs or {}
        # entity_representations_kwargs.pop("triples_factory", None)
        # note: this is *not* a nn.ModuleDict; the modules have to be registered elsewhere
        _mode_to_representations = {TRAINING: self.entity_representations}
        if "triples_factory" in entity_representations_kwargs:
            entity_representations_kwargs = ChainMap(
                dict(triples_factory=validation_factory), entity_representations_kwargs
            )
        _mode_to_representations[VALIDATION] = validation_entity_representations = self._build_representations(
            triples_factory=validation_factory,
            representations=entity_representations,
            representations_kwargs=entity_representations_kwargs,
            label="entity",
        )

        # shared
        if testing_factory is None:
            testing_entity_representations = validation_entity_representations
        else:
            # non-shared
            if "triples_factory" in entity_representations_kwargs:
                entity_representations_kwargs = ChainMap(
                    dict(triples_factory=testing_factory), entity_representations_kwargs
                )
            testing_entity_representations = self._build_representations(
                triples_factory=testing_factory,
                representations=entity_representations,
                representations_kwargs=entity_representations_kwargs,
                label="entity",
            )
        _mode_to_representations[TESTING] = testing_entity_representations
        # note: "training" is an attribute of nn.Module -> need to rename to avoid name collision
        self._mode_to_representations = nn.ModuleDict({f"{k}_factory": v for k, v in _mode_to_representations.items()})

    # docstr-coverage: inherited
    def _get_entity_representations_from_inductive_mode(
        self, *, mode: Optional[InductiveMode]
    ) -> Sequence[Representation]:  # noqa: D102
        if mode is None:
            raise ValueError(
                f"{self.__class__.__name__} does not support the transductive setting (i.e., when mode is None)"
            )
        key = f"{mode}_factory"
        if key in self._mode_to_representations:
            return self._mode_to_representations[key]
        raise ValueError(f"{self.__class__.__name__} does not support mode={mode}")

    # docstr-coverage: inherited
    def _get_entity_len(self, *, mode: Optional[InductiveMode]) -> Optional[int]:  # noqa: D102
        return self._get_entity_representations_from_inductive_mode(mode=mode)[0].max_id
