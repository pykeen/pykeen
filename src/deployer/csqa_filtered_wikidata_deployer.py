# -*- coding: utf-8 -*-
import os
import pickle
import sys
import time

from utilities.constants import KG_EMBEDDINGS_PIPELINE_DIR, ENTITY_TO_EMBEDDINGS, EVAL_RESULTS, CSQA_WIKIDATA

w_dir = os.path.dirname(os.getcwd())
sys.path.append(w_dir)

import click
import yaml
from utilities.pipeline import Pipeline


@click.command()
@click.option('-cfg_path', help='path to config file', required=True)
def main(cfg_path):
    with open(cfg_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    pipeline = Pipeline(config=cfg)

    trained_kg_model, eval_summary, entity_to_embedding, relation_to_embedding = pipeline.start_pipeline(
        learning_rate=0.01, num_epochs=1,
        ratio_of_neg_triples=0.5,
        batch_size=None,
        ratio_test_data=1 / 5, seed=2)

    current_time = time.strftime("%H:%M:%S")
    current_date = time.strftime("%d/%m/%Y").replace('/', '-')
    entity_to_embedding_out = os.path.join(KG_EMBEDDINGS_PIPELINE_DIR, CSQA_WIKIDATA, 'output', current_date,
                                           current_time)
    os.makedirs(entity_to_embedding_out, exist_ok=True)

    eval_results_out = os.path.join(KG_EMBEDDINGS_PIPELINE_DIR, CSQA_WIKIDATA, 'output', current_date, current_time)

    os.makedirs(eval_results_out, exist_ok=True)

    entity_to_embedding_out = os.path.join(entity_to_embedding_out, ENTITY_TO_EMBEDDINGS + '.pkl')
    eval_results_out = os.path.join(eval_results_out, EVAL_RESULTS + '.pkl')

    with open(entity_to_embedding_out, 'wb') as handle:
        pickle.dump(entity_to_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(eval_results_out, 'wb') as handle:
        pickle.dump(eval_summary, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
