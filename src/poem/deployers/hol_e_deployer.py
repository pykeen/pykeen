import json
import logging
import os
import time

import click
import numpy as np
import torch
from poem.models.unimodal.hole import HolE
from torch import optim

from poem.constants import GPU, KG_EMBEDDING_MODEL_NAME, EMBEDDING_DIM, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS
from poem.evaluation import RankBasedEvaluator
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.preprocessing.triples_preprocessing_utils.basic_triple_utils import create_entity_and_relation_mappings, \
    load_triples, map_triples_elements_to_ids
from poem.training_loops import OWATrainingLoop

log = logging.getLogger(__name__)


@click.command()
@click.option('-training', '--training_file')
@click.option('-test', '--test_file')
@click.option('-out', '--output_direc')
def main(training_file, test_file, output_direc):
    """"""

    output_directory = os.path.join(output_direc, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.mkdir(output_directory)

    # Step 1: Create instances
    log.info("Create instances")
    training_triples = load_triples(path=training_file)
    test_triples = load_triples(path=test_file)
    all_triples = np.concatenate([training_triples, test_triples], axis=0)

    entity_to_id, relation_to_id = create_entity_and_relation_mappings(triples=all_triples)
    mapped_training_triples = map_triples_elements_to_ids(triples=training_triples,
                                                          entity_to_id=entity_to_id,
                                                          rel_to_id=relation_to_id)
    factory = TriplesFactory(entity_to_id=entity_to_id,
                             relation_to_id=relation_to_id)

    instances = factory.create_owa_instances(triples=training_triples)

    embedding_dim = 20
    num_negs_per_pos = 1
    learning_rate = 0.1
    batch_size = 200
    num_epochs = 10

    # Step 2: Configure KGE model
    kge_model = HolE(num_entities=len(entity_to_id),
                        num_relations=len(relation_to_id),
                        embedding_dim=embedding_dim,
                        criterion=torch.nn.MarginRankingLoss(margin=1., reduction='mean'),
                        preferred_device=GPU)
    kge_model = kge_model.to(kge_model.device)

    parameters = filter(lambda p: p.requires_grad, kge_model.parameters())

    optimizer = optim.Adagrad(params=parameters, lr=learning_rate)

    # Step 3: Train
    all_entities = np.array(list(entity_to_id.values()), dtype=np.long)
    log.info("Train KGE model")

    owa_training_loop = OWATrainingLoop(kge_model=kge_model,
                                        optimizer=optimizer,
                                        all_entities=all_entities)

    fitted_kge_model, losses = owa_training_loop.train(training_instances=instances,
                                                       num_epochs=num_epochs,
                                                       batch_size=batch_size,
                                                       num_negs_per_pos=num_negs_per_pos
                                                       )

    # Step 4: Prepare test triples
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
    metric_results = evaluator.evaluate(test_triples=mapped_test_triples[0:100, :])

    # Step 7: Create summary
    config = {
        KG_EMBEDDING_MODEL_NAME: kge_model.model_name,
        EMBEDDING_DIM: embedding_dim,
        LEARNING_RATE: learning_rate,
        BATCH_SIZE: batch_size,
        NUM_EPOCHS: num_epochs
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
