# -*- coding: utf-8 -*-

import json
import logging
import os
import time

import click
import numpy as np
from torch import optim

from poem.evaluation import RankBasedEvaluator
from poem.hyper_parameter_optimization.random_search import RandomSearch
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.models import TransE
from poem.preprocessing.triples_preprocessing_utils.basic_triple_utils import (
    create_entity_and_relation_mappings,
    load_triples, map_triples_elements_to_ids,
)
from poem.training_loops import OWATrainingLoop

log = logging.getLogger(__name__)


@click.command()
@click.option('-training', '--training_file')
@click.option('-test', '--test_file')
@click.option('-out', '--output_direc')
def main(training_file, test_file, output_direc):
    output_directory = os.path.join(output_direc, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.mkdir(output_directory)

    # Step 1: Create instances
    log.info("Create instances")
    training_triples = load_triples(path=training_file)

    entity_to_id, relation_to_id = create_entity_and_relation_mappings(triples=training_triples)
    mapped_training_triples = map_triples_elements_to_ids(
        triples=training_triples,
        entity_to_id=entity_to_id,
        rel_to_id=relation_to_id,
    )
    factory = TriplesFactory(
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )

    training_instances = factory.create_owa_instances(triples=training_triples)

    params_to_values = {
        'embedding_dim': [50, 100],
        'learning_rate': [0.1, 0.01],
        'batch_size': 32,
        'num_epochs': 1,
        'num_entities': len(entity_to_id),
        'num_relations': len(relation_to_id),

    }

    all_entities = np.array(list(entity_to_id.values()), dtype=np.long)
    owa_training_loop = OWATrainingLoop(all_entities=all_entities)

    test_triples = load_triples(path=test_file)
    mapped_test_triples = map_triples_elements_to_ids(
        triples=test_triples,
        entity_to_id=entity_to_id,
        rel_to_id=relation_to_id,
    )

    # Configure evaluator
    evaluator = RankBasedEvaluator(
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        training_triples=mapped_training_triples,
        filter_neg_triples=False,
    )

    rs = RandomSearch(
        model_class=TransE,
        optimizer_class=optim.SGD,
        entity_to_id=entity_to_id,
        rel_to_id=relation_to_id,
        training_loop=owa_training_loop,
        evaluator=evaluator,
    )

    # Apply random search
    trained_model, losses, metric_result, model_params = rs.optimize_hyperparams(
        training_instances=training_instances,
        test_triples=mapped_test_triples,
        params_to_values=params_to_values,
    )

    # Create summary
    eval_file = os.path.join(output_directory, 'evaluation_summary.json')

    with open(eval_file, 'w') as file:
        json.dump(metric_result.to_json(), file, indent=2)

    losses_file = os.path.join(output_directory, 'losses.json')

    with open(losses_file, 'w') as file:
        json.dump(losses, file, indent=2)

    config_file = os.path.join(output_directory, 'config.json')

    with open(config_file, 'w') as file:
        json.dump(model_params, file, indent=2)


if __name__ == '__main__':
    main()
