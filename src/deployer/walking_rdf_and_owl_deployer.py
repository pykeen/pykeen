# -*- coding: utf-8 -*-
import os
import sys

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


    pipeline = Pipeline(config=cfg, seed=2)

    trained_model, eval_summary, entity_to_embedding, relation_to_embedding = pipeline.start_training()

    print(eval_summary)


if __name__ == '__main__':
    main()
