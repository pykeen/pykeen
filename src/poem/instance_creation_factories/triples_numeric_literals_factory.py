# -*- coding: utf-8 -*-

"""Implementation of factory that create instances containing of triples and numeric literals."""
import numpy as np

from kupp.numeric_literals_preprocessing_utils.basic_utils import create_matix_of_literals
from kupp.triples_preprocessing_utils.basic_triple_utils import load_triples
from poem.constants import PATH_TO_NUMERIC_LITERALS, NUMERIC_LITERALS
from poem.instance_creation_factories.triples_factory import TriplesFactory, Instances


class TriplesNumericLiteralsFactory(TriplesFactory):
    """."""

    def __init__(self, config):
        # FIXME: Return error: TypeError: object.__init__() takes no arguments
        # super(TriplesFactory, self).__init__(config)
        self.config = config

    def _create_numeric_literals(self) -> np.ndarray:
        """"""
        numeric_triples = load_triples(path=self.config[PATH_TO_NUMERIC_LITERALS], delimiter='\t')
        numeric_literals = create_matix_of_literals(numeric_triples=numeric_triples, entity_to_id=self.entity_to_id)
        return numeric_literals

    def _add_nummerical_literals(self, instances: Instances, numeric_literals) -> Instances:
        """"""

        instances.multimodal_data = {
            NUMERIC_LITERALS: numeric_literals
        }
        instances.has_multimodal_data = True

        return instances

    def create_train_and_test_instances(self) -> (Instances, Instances):
        """"""
        train_instances, test_instances = super().create_train_and_test_instances()
        numeric_literals = self._create_numeric_literals()

        train_instances = self._add_nummerical_literals(instances=train_instances, numeric_literals=numeric_literals)
        test_instances = self._add_nummerical_literals(instances=test_instances, numeric_literals=numeric_literals)

        return train_instances, test_instances

    def create_instances(self) -> Instances:
        """"""
        triple_instances = super().create_instances()
        numeric_literals = self._create_numeric_literals()
        triple_instances = self._add_nummerical_literals(instances=triple_instances, numeric_literals=numeric_literals)
        return triple_instances
