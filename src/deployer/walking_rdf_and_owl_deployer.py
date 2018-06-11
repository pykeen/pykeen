# -*- coding: utf-8 -*-
import click
import yaml

from utilities.pipeline import Pipeline


@click.command()
@click.option('-cfg_path', help='path to config file', required=True)
def main(cfg_path):
    with open(cfg_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    pipeline = Pipeline(config=cfg)

    trained_kg_model, eval_summary = pipeline.start_pipeline(learning_rate=0.001, num_epochs=10,
                                                             ratio_of_neg_triples=0.5,
                                                             batch_size=None, ratio_test_data=1/5, seed=2)

    print(eval_summary)


if __name__ == '__main__':
    main()
