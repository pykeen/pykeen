import torch

from poem.constants import KG_ASSUMPTION, EMBEDDING_DIM, CWA, NUM_ENTITIES, NUM_RELATIONS
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.kge_models.unimodal_kge_models.complex_cwa import ComplexCWA
from poem.model_config import ModelConfig
from poem.preprocessing.triples_preprocessing_utils.basic_triple_utils import create_entity_and_relation_mappings, \
    load_triples
import numpy as np

if __name__ == '__main__':
    path_to_training_data = '/Users/mali/PycharmProjects/POEM/tests/resources/test.txt'

    # Step 1: Create instances
    training_triples = load_triples(path=path_to_training_data)
    entity_to_id, relation_to_id = create_entity_and_relation_mappings(triples=training_triples)
    factory = TriplesFactory(entity_to_id=entity_to_id, relation_to_id=relation_to_id)
    instances = factory.create_cwa_instances(triples=training_triples)

    # Step 2: Define config
    config = {
        KG_ASSUMPTION: CWA,
        EMBEDDING_DIM: 200,
        NUM_ENTITIES: len(entity_to_id),
        NUM_RELATIONS: len(relation_to_id)
    }

    model_config = ModelConfig(config=config, multimodal_data=None)
    kge_model = ComplexCWA(model_config=model_config)
