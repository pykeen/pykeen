# -*- coding: utf-8 -*-

"""Implementation of factory that create instances containing of triples and numeric literals."""
from kupp.numeric_literals_preprocessing_utils.basic_utils import create_matix_of_literals
from kupp.triples_preprocessing_utils.basic_triple_utils import load_triples
from poem.constants import PATH_TO_NUMERIC_LITERALS, NUMERIC_LITERALS
from poem.instance_creation_factories.triples_factory import TriplesFactory, Instances


class TriplesNumericLiteralsFactory(TriplesFactory):
    """."""

    def __int__(self, config):
        """."""
        self.config = config

    def create_instances(self) -> Instances:
        """"""
        triple_instances = super().create_instances()
        numeric_triples = load_triples(path=self.config[PATH_TO_NUMERIC_LITERALS], delimiter='\t')
        numeric_literals = create_matix_of_literals(numeric_triples=numeric_triples, entity_to_id=self.entity_to_id)
        instances = triple_instances
        instances.multimodal_data = {
            NUMERIC_LITERALS: numeric_literals
        }
        instances.has_multimodal_data = True
        return instances
