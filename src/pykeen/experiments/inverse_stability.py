"""Inverse Stability Workflow."""

import logging
from typing import Type

import click
import matplotlib.pyplot as plt
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
    for dataset in ['nations', 'kinships']:
        for model in ['rotate', 'complex', 'simple', 'transe', 'distmult']:
            run_inverse_stability_workflow(dataset=dataset, model=model)


def run_inverse_stability_workflow(dataset: str, model: str, random_seed=0, device='cpu'):
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
        random_seed=random_seed,
        device=device,
    )
    test_tf = dataset.testing
    model = pipeline_result.model
    # Score with original triples
    scores = model.score_hrt(test_tf.mapped_triples)
    scores_np = scores.detach().numpy()[:, 0]
    forward_path = dataset_dir / f'{model_name}_forward.txt'
    with forward_path.open('w') as file:
        for forward_score in scores_np:
            print(forward_score, file=file)

    # Score with inverse triples
    scores_inverse = model.score_hrt_inverse(test_tf.mapped_triples)
    scores_inverse_np = scores_inverse.detach().numpy()[:, 0]

    inverse_path = dataset_dir / f'{model_name}_inverse.txt'
    with inverse_path.open('w') as file:
        for inverse_score in scores_inverse_np:
            print(inverse_score, file=file)

    fig, ax = plt.subplots(1, 1)
    sns.histplot(scores_np, ax=ax, label='Forward', color='blue')
    sns.histplot(scores_inverse_np, ax=ax, label='Inverse', color='orange')
    ax.set_title(f'{dataset_name} - {model_name}')
    ax.set_xlabel('Score')
    plt.legend()
    plt.savefig(dataset_dir / f'{model_name}_overlay.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    sns.histplot(scores_np - scores_inverse_np, ax=ax)
    ax.set_title(f'{dataset_name} - {model_name}')
    ax.set_xlabel('Forward - Inverse Score Difference')
    plt.savefig(dataset_dir / f'{model_name}_residuals.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
