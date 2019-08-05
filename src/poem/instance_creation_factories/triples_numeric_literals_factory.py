# -*- coding: utf-8 -*-

"""Implementation of factory that create instances containing of triples and numeric literals.tsv."""

from typing import Optional, TextIO, Union

import numpy as np

from poem.constants import NUMERIC_LITERALS
from poem.instance_creation_factories.instances import MultimodalCWAInstances, MultimodalOWAInstances
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.preprocessing.instance_creation_utils import create_matrix_of_literals
from poem.preprocessing.utils import load_triples

__all__ = [
    'TriplesNumericLiteralsFactory',
]


class TriplesNumericLiteralsFactory(TriplesFactory):
    """Create multi-modal instances given the path to triples."""

    def __init__(
            self,
            *,
            path: Union[None, str, TextIO] = None,
            triples: Optional[np.ndarray] = None,
            path_to_numeric_triples: Union[None, str, TextIO] = None,
            numeric_triples: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the multi-modal triples factory.

        :param path:
        :param path_to_numeric_triples:
        """
        super().__init__(path=path, triples=triples)

        if path_to_numeric_triples is None and numeric_triples is None:
            raise ValueError('Must specify one of path_to_numeric_triples or numeric_triples')
        elif path_to_numeric_triples is not None and numeric_triples is not None:
            raise ValueError('Must not specify both path_to_numeric_triples and numeric_triples')
        elif path_to_numeric_triples is not None:
            self.path_to_numeric_triples = path_to_numeric_triples
            self.numeric_triples = load_triples(self.path_to_numeric_triples)
        else:  # numeric_triples is not None:
            self.path_to_numeric_triples = '<None>'
            self.numeric_triples = numeric_triples

        self.numeric_literals = None
        self.multimodal_data = None
        self.literals_to_id = None

        self._create_numeric_literals()

    def __repr__(self):  # noqa: D105
        return f'{self.__class__.__name__}(path="{self.path}", ' \
               f'path_to_numeric_triples="{self.path_to_numeric_triples}")'

    def _create_numeric_literals(self) -> None:
        self.numeric_literals, self.literals_to_id = create_matrix_of_literals(
            numeric_triples=self.numeric_triples,
            entity_to_id=self.entity_to_id,
        )
        self.multimodal_data = {
            NUMERIC_LITERALS: self.numeric_literals,
        }

    def create_owa_instances(self) -> MultimodalOWAInstances:
        """Create multi-modal OWA instances for this factory's triples."""
        owa_instances = super().create_owa_instances()

        if self.multimodal_data is None:
            self._create_numeric_literals()

        return MultimodalOWAInstances(
            instances=owa_instances.instances,
            entity_to_id=owa_instances.entity_to_id,
            relation_to_id=owa_instances.relation_to_id,
            kg_assumption=owa_instances.kg_assumption,
            multimodal_data=self.multimodal_data,
        )

    def create_cwa_instances(self) -> MultimodalCWAInstances:
        """Create multi-modal CWA instances for this factory's triples."""
        cwa_instances = super().create_cwa_instances()

        if self.multimodal_data is None:
            self._create_numeric_literals()

        return MultimodalCWAInstances(
            instances=cwa_instances.instances,
            entity_to_id=cwa_instances.entity_to_id,
            relation_to_id=cwa_instances.relation_to_id,
            kg_assumption=cwa_instances.kg_assumption,
            multimodal_data=self.multimodal_data,
            data_relation_to_id=self.literals_to_id,
            labels=cwa_instances.labels,
        )
