# -*- coding: utf-8 -*-

import json
import logging
import os
import time

import click
import numpy as np
from torch import nn, optim

from poem.constants import BATCH_SIZE, EMBEDDING_DIM, GPU, LEARNING_RATE, MODEL_NAME, NUM_EPOCHS
from poem.evaluation import RankBasedEvaluator
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.models.unimodal.distmult import DistMult
from poem.preprocessing.triples_preprocessing_utils.basic_triple_utils import (
    create_entity_and_relation_mappings,
    load_triples, map_triples_elements_to_ids,
)
from poem.training_loops import OWATrainingLoop
from poem.utils import get_params_requiring_grad

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

    instances = factory.create_owa_instances(triples=training_triples)

    embedding_dim = 50
    learning_rate = 0.01
    batch_size = 32
    num_epochs = 1

    # Step 2: Configure KGE model
    model = DistMult(
        num_entities=len(entity_to_id),
        num_relations=len(relation_to_id),
        embedding_dim=embedding_dim,
        criterion=nn.MarginRankingLoss(margin=1., reduction='mean'),
        preferred_device=GPU,
    )

    params = get_params_requiring_grad(model)
    optimizer = optim.SGD(params=params, lr=learning_rate)

    # Step 3: Train
    all_entities = np.array(list(entity_to_id.values()), dtype=np.long)
    log.info("Train KGE model")

    owa_training_loop = OWATrainingLoop(
        model=model,
        optimizer=optimizer,
        all_entities=all_entities,
    )

    _, losses = owa_training_loop.train(
        training_instances=instances,
        num_epochs=num_epochs,
        batch_size=batch_size,
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
    metric_results = evaluator.evaluate(test_triples=mapped_test_triples[0:100, :])

    # Step 7: Create summary
    config = {
        MODEL_NAME: model.model_name,
        EMBEDDING_DIM: embedding_dim,
        LEARNING_RATE: learning_rate,
        BATCH_SIZE: batch_size,
        NUM_EPOCHS: num_epochs,
    }

    eval_file = os.path.join(output_directory, 'evaluation_summary.json')

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
