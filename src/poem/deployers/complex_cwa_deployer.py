# -*- coding: utf-8 -*-

from torch import optim

from poem.constants import EMBEDDING_DIM, INPUT_DROPOUT, NUM_ENTITIES, NUM_RELATIONS
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.models import ComplexCWA
from poem.preprocessing.triples_preprocessing_utils.basic_triple_utils import (
    create_entity_and_relation_mappings, load_triples,
)
from poem.training_loops import CWATrainingLoop
from poem.utils import get_params_requiring_grad

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
    model = ComplexCWA(**config)
    params = get_params_requiring_grad(model)
    optimizer = optim.Adam(params=params)

    # Train
    cwa_training_loop = CWATrainingLoop(model=model, optimizer=optimizer)

    _, losses = cwa_training_loop.train(
        training_instances=instances,
        num_epochs=2,
        batch_size=128,
        label_smoothing=True,
        label_smoothing_epsilon=0.1,
    )
