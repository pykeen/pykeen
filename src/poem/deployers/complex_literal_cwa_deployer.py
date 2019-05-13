from poem.evaluation.ranked_based_evaluator import RankBasedEvaluator
from poem.instance_creation_factories.triples_numeric_literals_factory import TriplesNumericLiteralsFactory
from poem.kge_models.kge_models_using_numerical_literals.complex_literal_cwa import ComplexLiteralCWA
from poem.preprocessing.triples_preprocessing_utils.basic_triple_utils import create_entity_and_relation_mappings, \
    load_triples, map_triples_elements_to_ids
from torch import optim
import torch
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

    # Step 2: Configure KGE model
    kge_model = ComplexLiteralCWA(embedding_dim=200,
                                  num_entities=len(entity_to_id),
                                  num_relations=len(relation_to_id),
                                  input_dropout=0.2,
                                  multimodal_data=instances.multimodal_data)

    parameters = filter(lambda p: p.requires_grad, kge_model.parameters())
    optimizer = optim.Adam(params=parameters)

    # Step 3: Train
    cwa_training_loop = CWATrainingLoop(kge_model=kge_model, optimizer=optimizer)

    fitted_kge_model, losses = cwa_training_loop.train(training_instances=instances,
                                                       num_epochs=2,
                                                       batch_size=128,
                                                       label_smoothing=True,
                                                       label_smoothing_epsilon=0.1
                                                       )

    # Step 4: Prepare test triples
    mapped_test_triples = map_triples_elements_to_ids(triples=training_triples[0:100, :],
                                                      entity_to_id=entity_to_id,
                                                      rel_to_id=relation_to_id)

    mapped_test_triples = torch.tensor(mapped_test_triples, dtype=torch.long, device=fitted_kge_model.device)


    # Step 5: Predict
    predictions = fitted_kge_model.predict(mapped_test_triples)

    # Step 6: Configure evaluator
    mapped_training_triples = map_triples_elements_to_ids(triples=training_triples,
                                                          entity_to_id=entity_to_id,
                                                          rel_to_id=relation_to_id)

    evaluator = RankBasedEvaluator(kge_model=fitted_kge_model,
                                   entity_to_id=entity_to_id,
                                   relation_to_id=relation_to_id,
                                   training_triples=mapped_test_triples,
                                   filter_neg_triples=False)

    # Step 7: Evaluate
    metric_results = evaluator.evaluate(test_triples=mapped_test_triples)

    print(metric_results)

