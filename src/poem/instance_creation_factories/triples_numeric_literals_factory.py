# -*- coding: utf-8 -*-

"""Implementation of factory that create instances containing of triples and numeric literals."""

from typing import Tuple

import numpy as np
from typing import Dict
from poem.constants import PATH_TO_NUMERIC_LITERALS, NUMERIC_LITERALS
from poem.instance_creation_factories.instances import MultimodalOWAInstances, \
    MultimodalCWAInstances
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.preprocessing.numeric_literals_preprocessing_utils.basic_utils import create_matix_of_literals


class TriplesNumericLiteralsFactory(TriplesFactory):
    """."""

    def __init__(self, entity_to_id, relation_to_id, numeric_triples):
        super().__init__(entity_to_id, relation_to_id)
        self.numeric_triples = numeric_triples
        self.numeric_literals = None
        self.multimodal_data = None

    def _create_numeric_literals(self) -> np.ndarray:
        """"""
        self.numeric_literals = create_matix_of_literals(numeric_triples=self.numeric_triples,
                                                         entity_to_id=self.entity_to_id)
        self.multimodal_data = {
            NUMERIC_LITERALS: self.numeric_literals
        }

    def create_owa_instances(self, triples) -> MultimodalOWAInstances:
        """"""
        owa_instances = super().create_owa_instances(triples=triples)

        if self.multimodal_data is None:
            self._create_numeric_literals()

        return MultimodalOWAInstances(instances=owa_instances.instances,
                                      entity_to_id=owa_instances.entity_to_id,
                                      relation_to_id=owa_instances.relation_to_id,
                                      kg_assumption=owa_instances.kg_assumption,
                                      multimodal_data=self.multimodal_data)

    def create_cwa_instances(self, triples) -> MultimodalCWAInstances:
        """."""
        cwa_instances = super().create_cwa_instances(triples=triples)

        if self.multimodal_data is None:
            self._create_numeric_literals()

        return MultimodalCWAInstances(instances=cwa_instances.instances,
                                      entity_to_id=cwa_instances.entity_to_id,
                                      relation_to_id=cwa_instances.relation_to_id,
                                      kg_assumption=cwa_instances.kg_assumption,
                                      multimodal_data=self.multimodal_data,
                                      labels=cwa_instances.labels)
