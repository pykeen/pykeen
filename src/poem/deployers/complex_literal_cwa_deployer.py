# -*- coding: utf-8 -*-

import json
import logging
import os
import time

import click
from torch import optim

from poem.constants import BATCH_SIZE, EMBEDDING_DIM, INPUT_DROPOUT, MODEL_NAME, LEARNING_RATE, NUM_EPOCHS
from poem.evaluation import RankBasedEvaluator
from poem.instance_creation_factories.triples_numeric_literals_factory import TriplesNumericLiteralsFactory
from poem.models import ComplexLiteralCWA
from poem.preprocessing.triples_preprocessing_utils.basic_triple_utils import (
    create_entity_and_relation_mappings,
    load_triples, map_triples_elements_to_ids,
)
from poem.training_loops import CWATrainingLoop
from poem.utils import get_params_requiring_grad

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
    mapped_training_triples = map_triples_elements_to_ids(
        triples=training_triples,
        entity_to_id=entity_to_id,
        rel_to_id=relation_to_id,
    )
    factory = TriplesNumericLiteralsFactory(
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        numeric_triples=literals,
    )
    instances = factory.create_cwa_instances(triples=training_triples)

    embedding_dim = 10
    input_dropout = 0.2
    learning_rate = 0.001
    batch_size = 128
    num_epochs = 1

    # Step 2: Configure KGE model
    model = ComplexLiteralCWA(
        embedding_dim=embedding_dim,
        num_entities=len(entity_to_id),
        num_relations=len(relation_to_id),
        input_dropout=input_dropout,
        multimodal_data=instances.multimodal_data,
    )

    params = get_params_requiring_grad(model)
    optimizer = optim.Adam(params=params, lr=learning_rate)

    # Step 3: Train
    log.info("Train KGE model")
    cwa_training_loop = CWATrainingLoop(model=model, optimizer=optimizer)

    _, losses = cwa_training_loop.train(
        training_instances=instances,
        num_epochs=num_epochs,
        batch_size=batch_size,
        label_smoothing=True,
        label_smoothing_epsilon=0.1,
    )

    # Step 4: Prepare test triples
    test_triples = load_triples(path=test_file)
    mapped_test_triples = map_triples_elements_to_ids(
        triples=test_triples,
        entity_to_id=entity_to_id,
        rel_to_id=relation_to_id,
    )

    # Step 5: Configure evaluator
    log.info("Evaluate KGE model")
    evaluator = RankBasedEvaluator(
        model=model,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        training_triples=mapped_training_triples,
        filter_neg_triples=False,
    )

    # Step 6: Evaluate
    metric_results = evaluator.evaluate(test_triples=mapped_test_triples)

    eval_file = os.path.join(output_directory, 'evaluation_summary.json')

    # Step 7: Create summary
    config = {
        MODEL_NAME: model.model_name,
        EMBEDDING_DIM: embedding_dim,
        INPUT_DROPOUT: input_dropout,
        LEARNING_RATE: learning_rate,
        BATCH_SIZE: batch_size,
        NUM_EPOCHS: num_epochs
    }

    with open(eval_file, 'w') as file:
        json.dump(metric_results.to_json(), file, indent=2)

    losses_file = os.path.join(output_directory, 'losses.json')

    with open(losses_file, 'w') as file:
        json.dump(losses, file, indent=2)

    config_file = os.path.join(output_directory, 'config.json')

    with open(config_file, 'w') as file:
        json.dump(config, file, indent=2)


if __name__ == '__main__':
    main()
