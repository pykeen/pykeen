# -*- coding: utf-8 -*-

"""Implementation of factory that create instances containing of triples and numeric literals.tsv."""

from poem.constants import NUMERIC_LITERALS
from poem.instance_creation_factories.instances import MultimodalCWAInstances, MultimodalOWAInstances
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.preprocessing.numeric_literals_preprocessing_utils.basic_utils import create_matrix_of_literals
from poem.preprocessing.triples_preprocessing_utils.basic_triple_utils import load_triples


class TriplesNumericLiteralsFactory(TriplesFactory):

    def __init__(self, path_to_triples: str, path_to_numeric_triples: str) -> None:
        super().__init__(path_to_triples=path_to_triples)
        self.numeric_triples = load_triples(path_to_numeric_triples)
        self.numeric_literals = None
        self.multimodal_data = None
        self.literals_to_id = None
        self._create_numeric_literals()

    def _create_numeric_literals(self) -> None:
        self.numeric_literals, self.literals_to_id = create_matrix_of_literals(
            numeric_triples=self.numeric_triples,
            entity_to_id=self.entity_to_id,
        )
        self.multimodal_data = {
            NUMERIC_LITERALS: self.numeric_literals,
        }

    def create_owa_instances(self) -> MultimodalOWAInstances:
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
