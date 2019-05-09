from poem.instance_creation_factories.triples_numeric_literals_factory import TriplesNumericLiteralsFactory
from poem.kge_models.kge_models_using_numerical_literals.complex_literal_cwa import ComplexLiteralCWA
from poem.preprocessing.triples_preprocessing_utils.basic_triple_utils import create_entity_and_relation_mappings, \
    load_triples
from torch import optim
import numpy as np

from poem.training_loops.cwa_training_loop import CWATrainingLoop

if __name__ == '__main__':
    path_to_training_data = '../../../tests/resources/test.txt'
    path_to_literals = '../../../tests/resources/numerical_literals.txt'

    # Step 1: Create instances
    training_triples = load_triples(path=path_to_training_data)
    literals = load_triples(path=path_to_literals)
    entity_to_id, relation_to_id = create_entity_and_relation_mappings(triples=training_triples)
    factory = TriplesNumericLiteralsFactory(entity_to_id=entity_to_id,
                                            relation_to_id=relation_to_id,
                                            numeric_triples=literals)
    instances = factory.create_cwa_instances(triples=training_triples)

    # Configure KGE model
    kge_model = ComplexLiteralCWA(embedding_dim=200,
                                  num_entities=len(entity_to_id),
                                  num_relations=len(relation_to_id),
                                  input_dropout=0.2,
                                  multimodal_data=instances.multimodal_data)

    parameters = filter(lambda p: p.requires_grad, kge_model.parameters())
    optimizer = optim.Adam(params=parameters)

    all_entities = np.array(list(entity_to_id.values()))

    # Train
    cwa_training_loop = CWATrainingLoop(kge_model=kge_model, optimizer=optimizer, all_entities=all_entities)

    fitted_kge_model, losses = cwa_training_loop.train(training_instances=instances,
                                                       num_epochs=2,
                                                       batch_size=128)

    print(losses)
