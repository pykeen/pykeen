from poem.constants import EMBEDDING_DIM, NUM_ENTITIES, NUM_RELATIONS, INPUT_DROPOUT
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.kge_models.unimodal_kge_models.complex_cwa import ComplexCWA
from poem.preprocessing.triples_preprocessing_utils.basic_triple_utils import create_entity_and_relation_mappings, \
    load_triples
from torch import optim
import numpy as np

from poem.training_loops.cwa_training_loop import CWATrainingLoop

if __name__ == '__main__':
    path_to_training_data = '../../../tests/resources/test.txt'

    # Step 1: Create instances
    training_triples = load_triples(path=path_to_training_data)
    entity_to_id, relation_to_id = create_entity_and_relation_mappings(triples=training_triples)
    factory = TriplesFactory(entity_to_id=entity_to_id, relation_to_id=relation_to_id)
    instances = factory.create_cwa_instances(triples=training_triples)

    # Step 2: Define config
    config = {
        EMBEDDING_DIM: 200,
        NUM_ENTITIES: len(entity_to_id),
        NUM_RELATIONS: len(relation_to_id),
        INPUT_DROPOUT: 0.2
    }

    # Configure KGE model
    kge_model = ComplexCWA(**config)
    parameters = filter(lambda p: p.requires_grad, kge_model.parameters())
    optimizer = optim.Adam(params=parameters)

    all_entities = np.array(list(entity_to_id.values()))

    # Train
    cwa_training_loop = CWATrainingLoop(kge_model=kge_model, optimizer=optimizer, all_entities=all_entities)

    fitted_kge_model, losses = cwa_training_loop.train(training_instances=instances,
                                                       num_epochs=2,
                                                       batch_size=128)
