# -*- coding: utf-8 -*-

"""Implementation of factory that create instances containing of triples and numeric literals."""

from typing import Tuple

import numpy as np
from typing import Dict
from kupp.numeric_literals_preprocessing_utils.basic_utils import create_matix_of_literals
from kupp.triples_preprocessing_utils.basic_triple_utils import load_triples
from poem.constants import PATH_TO_NUMERIC_LITERALS, NUMERIC_LITERALS, OWA, CWA
from poem.instance_creation_factories.instances import MultimodalInstances, MultimodalOWAInstances, \
    MultimodalCWAInstances
from poem.instance_creation_factories.triples_factory import TriplesFactory, Instances


class TriplesNumericLiteralsFactory(TriplesFactory):
    """."""

    def __init__(self, config: Dict):
        super().__init__(config)

    def _create_numeric_literals(self) -> np.ndarray:
        """"""
        numeric_triples = load_triples(path=self.config[PATH_TO_NUMERIC_LITERALS], delimiter='\t')
        numeric_literals = create_matix_of_literals(numeric_triples=numeric_triples, entity_to_id=self.entity_to_id)
        return numeric_literals

    def _create_multimodal_instances(self, instances: Instances, numeric_literals) -> MultimodalInstances:
        """"""

        multimodal_data = {
            NUMERIC_LITERALS: numeric_literals
        }

        if instances.kg_assumption == OWA:
            return MultimodalOWAInstances(instances=instances.instances,
                                          entity_to_id=instances.entity_to_id,
                                          relation_to_id=instances.relation_to_id,
                                          kg_assumption=instances.kg_assumption,
                                          multimodal_data=multimodal_data)
        elif instances.kg_assumption == CWA:
            return MultimodalCWAInstances(instances=instances.instances,
                                          entity_to_id=instances.entity_to_id,
                                          relation_to_id=instances.relation_to_id,
                                          kg_assumption=instances.kg_assumption,
                                          multimodal_data=multimodal_data)

    def create_train_and_test_instances(self) -> Tuple[Instances, Instances]:
        """"""
        train_instances, test_instances = super().create_train_and_test_instances()
        numeric_literals = self._create_numeric_literals()

        train_instances = self._create_multimodal_instances(instances=train_instances,
                                                            numeric_literals=numeric_literals)
        test_instances = self._create_multimodal_instances(instances=test_instances,
                                                           numeric_literals=numeric_literals)

        return train_instances, test_instances

    def create_instances(self) -> Instances:
        """"""
        triple_instances = super().create_instances()
        numeric_literals = self._create_numeric_literals()
        return self._create_multimodal_instances(instances=triple_instances, numeric_literals=numeric_literals)
