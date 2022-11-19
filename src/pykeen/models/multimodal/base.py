# -*- coding: utf-8 -*-

"""Base classes for multi-modal models."""

from typing import ClassVar, Tuple, Type

import torch
from class_resolver import HintOrType, OneOrManyHintOrType, OneOrManyOptionalKwargs, OptionalKwargs

from ..nbase import ERModel
from ...nn.combination import Combination
from ...nn.init import PretrainedInitializer
from ...nn.modules import Interaction
from ...nn.representation import CombinedRepresentation, Embedding, Representation
from ...triples import TriplesNumericLiteralsFactory
from ...utils import upgrade_to_sequence

__all__ = [
    "LiteralModel",
]


class LiteralModel(
    ERModel[
        Tuple[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]
    ],
    autoreset=False,
):
    """Base class for models with entity literals that uses combinations from :class:`pykeen.nn.combinations`."""

    #: the interaction class (for generating the overview table)
    interaction_cls: ClassVar[Type[Interaction]]

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        interaction: HintOrType[Interaction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]],
        entity_representations: OneOrManyHintOrType[Representation] = None,
        entity_representations_kwargs: OneOrManyOptionalKwargs = None,
        combination: HintOrType[Combination] = None,
        combination_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        """
        Initialize the model.

        :param triples_factory:
            the (training) triples factory
        :param interaction:
            the interaction function
        :param entity_representations:
            the entity representations (excluding the ones from literals)
        :param entity_representations_kwargs:
            the entity representations keyword-based parameters (excluding the ones from literals)
        :param combination:
            the combination for entity and literal representations
        :param combination_kwargs:
            keyword-based parameters for instantiating the combination
        :param kwargs:
            additional keyword-based parameters passed to :meth:`ERModel.__init__`
        """
        literals = triples_factory.get_numeric_literals_tensor()
        _max_id, *shape = literals.shape
        entity_representations = tuple(upgrade_to_sequence(entity_representations)) + (Embedding,)
        entity_representations_kwargs = tuple(upgrade_to_sequence(entity_representations_kwargs)) + (
            dict(
                shape=shape,
                initializer=PretrainedInitializer(tensor=literals),
                trainable=False,
            ),
        )
        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            entity_representations=CombinedRepresentation,
            entity_representations_kwargs=dict(
                # added by ERModel
                # max_id=triples_factory.num_entities,
                base=entity_representations,
                base_kwargs=entity_representations_kwargs,
                combination=combination,
                combination_kwargs=combination_kwargs,
            ),
            **kwargs,
        )
