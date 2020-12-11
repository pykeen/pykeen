"""Inverse Stability Workflow."""

import itertools as itt
import logging
from typing import Type

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import pykeen.evaluation.evaluator
from pykeen.constants import PYKEEN_EXPERIMENTS
from pykeen.datasets import Dataset, get_dataset
from pykeen.models import Model, get_model_cls
from pykeen.pipeline import pipeline

INVERSE_STABILITY = PYKEEN_EXPERIMENTS / 'inverse_stability'
INVERSE_STABILITY.mkdir(parents=True, exist_ok=True)

pykeen.evaluation.evaluator.logger.setLevel(logging.CRITICAL)


@click.command()
def main():
    """Run the inverse stability experiments."""
    outer_dfs = []
    for dataset in ['nations', 'kinships']:
        inner_dfs = []
        for model in ['rotate', 'complex', 'simple', 'transe', 'distmult']:
            df = run_inverse_stability_workflow(dataset=dataset, model=model)
            inner_dfs.append(df)
            outer_dfs.append(df)
        inner_df = pd.concat(inner_dfs)
        inner_df.to_csv(INVERSE_STABILITY / dataset / 'results.tsv', sep='\t', index=False)

    outer_df = pd.concat(outer_dfs)
    outer_df.to_csv(INVERSE_STABILITY / 'results.tsv', sep='\t', index=False)


def run_inverse_stability_workflow(dataset: str, model: str, random_seed=0, device='cpu'):
    """Run an inverse stability experiment."""
    dataset: Dataset = get_dataset(
        dataset=dataset,
        dataset_kwargs=dict(
            create_inverse_triples=True,
        ),
    )
    dataset_name = dataset.get_normalized_name()
    model_cls: Type[Model] = get_model_cls(model)
    model_name = model_cls.__name__.lower()

    dataset_dir = INVERSE_STABILITY / dataset_name
    dataset_dir.mkdir(exist_ok=True, parents=True)

    pipeline_result = pipeline(
        dataset=dataset,
        model=model,
        training_kwargs=dict(
            num_epochs=1000,
            use_tqdm_batch=False,
        ),
        stopper='early',
        stopper_kwargs=dict(patience=5, frequency=5),
        random_seed=random_seed,
        device=device,
    )
    test_tf = dataset.testing
    model = pipeline_result.model
    # Score with original triples
    scores_forward = model.score_hrt(test_tf.mapped_triples)
    scores_forward_np = scores_forward.detach().numpy()[:, 0]

    # Score with inverse triples
    scores_inverse = model.score_hrt_inverse(test_tf.mapped_triples)
    scores_inverse_np = scores_inverse.detach().numpy()[:, 0]

    scores_path = dataset_dir / f'{model_name}_scores.tsv'
    df = pd.DataFrame(
        list(zip(itt.repeat(dataset_name), itt.repeat(model_name), scores_forward_np, scores_inverse_np)),
        columns=['dataset', 'model', 'forward', 'inverse'],
    )
    df.to_csv(scores_path, sep='\t', index=False)

    fig, ax = plt.subplots(1, 1)
    sns.histplot(data=df, x='forward', label='Forward', ax=ax, color='blue')
    sns.histplot(data=df, x='inverse', label='Inverse', ax=ax, color='orange')
    ax.set_title(f'{dataset_name} - {model_name}')
    ax.set_xlabel('Score')
    plt.legend()
    plt.savefig(dataset_dir / f'{model_name}_overlay.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    sns.histplot(scores_forward_np - scores_inverse_np, ax=ax)
    ax.set_title(f'{dataset_name} - {model_name}')
    ax.set_xlabel('Forward - Inverse Score Difference')
    plt.savefig(dataset_dir / f'{model_name}_residuals.png', dpi=300)
    plt.close(fig)

    return df


if __name__ == '__main__':
    main()
