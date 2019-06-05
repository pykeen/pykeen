from poem.constants import KG_EMBEDDING_MODEL_NAME, EMBEDDING_DIM, INPUT_DROPOUT, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS
from poem.evaluation import RankBasedEvaluator

from poem.instance_creation_factories.triples_numeric_literals_factory import TriplesNumericLiteralsFactory
from poem.models import ComplexLiteralCWA
from poem.preprocessing.triples_preprocessing_utils.basic_triple_utils import create_entity_and_relation_mappings, \
    load_triples, map_triples_elements_to_ids
from torch import optim
import click
import json
import time
import os
from collections import OrderedDict
import logging

from poem.training_loops import CWATrainingLoop

log = logging.getLogger(__name__)


@click.command()
@click.option('-training', '--training_file')
@click.option('-lit', '--literals_file')
@click.option('-test', '--test_file')
@click.option('-out', '--output_direc')
def main(training_file, test_file, output_direc, literals_file):
    """"""

    output_directory = os.path.join(output_direc, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.mkdir(output_directory)

    # Step 1: Create instances
    log.info("Create instances")
    training_triples = load_triples(path=training_file)

    literals = load_triples(path=literals_file)
    entity_to_id, relation_to_id = create_entity_and_relation_mappings(triples=training_triples)
    mapped_training_triples = map_triples_elements_to_ids(triples=training_triples,
                                                          entity_to_id=entity_to_id,
                                                          rel_to_id=relation_to_id)
    factory = TriplesNumericLiteralsFactory(entity_to_id=entity_to_id,
                                            relation_to_id=relation_to_id,
                                            numeric_triples=literals)
    instances = factory.create_cwa_instances(triples=training_triples)

    embedding_dim = 10
    input_dropout = 0.2
    learning_rate = 0.001
    batch_size = 128
    num_epochs = 1

    # Step 2: Configure KGE model
    kge_model = ComplexLiteralCWA(embedding_dim=embedding_dim,
                                  num_entities=len(entity_to_id),
                                  num_relations=len(relation_to_id),
                                  input_dropout=input_dropout,
                                  multimodal_data=instances.multimodal_data)

    parameters = filter(lambda p: p.requires_grad, kge_model.parameters())
    optimizer = optim.Adam(params=parameters, lr=learning_rate)

    # Step 3: Train
    log.info("Train KGE model")
    cwa_training_loop = CWATrainingLoop(kge_model=kge_model, optimizer=optimizer)

    fitted_kge_model, losses = cwa_training_loop.train(training_instances=instances,
                                                       num_epochs=num_epochs,
                                                       batch_size=batch_size,
                                                       label_smoothing=True,
                                                       label_smoothing_epsilon=0.1
                                                       )

    # Step 4: Prepare test triples
    test_triples = load_triples(path=test_file)
    mapped_test_triples = map_triples_elements_to_ids(triples=test_triples,
                                                      entity_to_id=entity_to_id,
                                                      rel_to_id=relation_to_id)

    # Step 5: Configure evaluator
    log.info("Evaluate KGE model")
    evaluator = RankBasedEvaluator(kge_model=fitted_kge_model,
                                   entity_to_id=entity_to_id,
                                   relation_to_id=relation_to_id,
                                   training_triples=mapped_training_triples,
                                   filter_neg_triples=False)

    # Step 6: Evaluate
    metric_results = evaluator.evaluate(test_triples=mapped_test_triples)
    results = OrderedDict()
    results['mean_rank'] = metric_results.mean_rank
    results['hits_at_k'] = metric_results.hits_at_k

    eval_file = os.path.join(output_directory, 'evaluation_summary.json')

    # Step 7: Create summary
    config = {
        KG_EMBEDDING_MODEL_NAME: kge_model.model_name,
        EMBEDDING_DIM: embedding_dim,
        INPUT_DROPOUT: input_dropout,
        LEARNING_RATE: learning_rate,
        BATCH_SIZE: batch_size,
        NUM_EPOCHS: num_epochs
    }


    with open(eval_file, 'w') as file:
        json.dump(results, file, indent=2)

    losses_file = os.path.join(output_directory, 'losses.json')

    with open(losses_file, 'w') as file:
        json.dump(losses, file, indent=2)

    config_file = os.path.join(output_directory, 'config.json')

    with open(config_file, 'w') as file:
        json.dump(config, file, indent=2)


if __name__ == '__main__':
    main()
