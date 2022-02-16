# -*- coding: utf-8 -*-

"""Inverse Stability Workflow.

This experiment investigates the differences between

"""

import itertools as itt
import logging
from typing import Optional, Type

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import pykeen.evaluation.evaluator
from pykeen.constants import PYKEEN_EXPERIMENTS
from pykeen.datasets import Dataset, get_dataset
from pykeen.models import Model, model_resolver
from pykeen.pipeline import pipeline
from pykeen.typing import InductiveMode

INVERSE_STABILITY = PYKEEN_EXPERIMENTS / "inverse_stability"
INVERSE_STABILITY.mkdir(parents=True, exist_ok=True)

pykeen.evaluation.evaluator.logger.setLevel(logging.CRITICAL)


@click.command()
@click.option("--force", is_flag=True)
@click.option("--clip", type=int, default=10)
@click.option("--mode")
def main(force: bool, clip: int, mode):
    """Run the inverse stability experiments."""
    results_path = INVERSE_STABILITY / "results.tsv"
    if results_path.exists() and not force:
        df = pd.read_csv(results_path, sep="\t")
        df["residuals"] = df["forward"] - df["inverse"]
        df = df[(-clip < df["residuals"]) & (df["residuals"] < clip)]
        g = sns.FacetGrid(df, col="model", row="dataset", hue="training_loop", sharex=False, sharey=False)
        g.map_dataframe(sns.histplot, x="residuals", stat="density")
        g.add_legend()
        g.savefig(INVERSE_STABILITY / "results_residuals.png", dpi=300)

    else:
        outer_dfs = []
        datasets = ["nations", "kinships"]
        models = ["rotate", "complex", "simple", "transe", "distmult"]
        training_loops = ["lcwa", "slcwa"]
        for dataset, model, training_loop in itt.product(datasets, models, training_loops):
            click.secho(f"{dataset} {model} {training_loop}", fg="cyan")
            df = run_inverse_stability_workflow(dataset=dataset, model=model, training_loop=training_loop, mode=mode)
            outer_dfs.append(df)
        outer_df = pd.concat(outer_dfs)
        outer_df.to_csv(INVERSE_STABILITY / "results.tsv", sep="\t", index=False)


def run_inverse_stability_workflow(
    dataset: str, model: str, training_loop: str, random_seed=0, device="cpu", *, mode: Optional[InductiveMode]
):
    """Run an inverse stability experiment."""
    dataset_instance: Dataset = get_dataset(
        dataset=dataset,
        dataset_kwargs=dict(
            create_inverse_triples=True,
        ),
    )
    dataset_name = dataset_instance.get_normalized_name()
    model_cls: Type[Model] = model_resolver.lookup(model)
    model_name = model_cls.__name__.lower()

    dataset_dir = INVERSE_STABILITY / dataset_name
    dataset_dir.mkdir(exist_ok=True, parents=True)

    pipeline_result = pipeline(
        dataset=dataset_instance,
        model=model,
        training_loop=training_loop,
        training_kwargs=dict(
            num_epochs=1000,
            use_tqdm_batch=False,
        ),
        stopper="early",
        stopper_kwargs=dict(patience=5, frequency=5),
        random_seed=random_seed,
        device=device,
    )
    test_tf = dataset_instance.testing
    model = pipeline_result.model
    # Score with original triples
    scores_forward = model.score_hrt(test_tf.mapped_triples, mode=mode)
    scores_forward_np = scores_forward.detach().numpy()[:, 0]

    # Score with inverse triples
    scores_inverse = model.score_hrt_inverse(test_tf.mapped_triples, mode=mode)
    scores_inverse_np = scores_inverse.detach().numpy()[:, 0]

    scores_path = dataset_dir / f"{model_name}_{training_loop}_scores.tsv"
    df = pd.DataFrame(
        list(
            zip(
                itt.repeat(training_loop),
                itt.repeat(dataset_name),
                itt.repeat(model_name),
                scores_forward_np,
                scores_inverse_np,
            )
        ),
        columns=["training_loop", "dataset", "model", "forward", "inverse"],
    )
    df.to_csv(scores_path, sep="\t", index=False)

    fig, ax = plt.subplots(1, 1)
    sns.histplot(data=df, x="forward", label="Forward", ax=ax, color="blue", stat="density")
    sns.histplot(data=df, x="inverse", label="Inverse", ax=ax, color="orange", stat="density")
    ax.set_title(f"{dataset_name} - {model_name} - {training_loop}")
    ax.set_xlabel("Score")
    plt.legend()
    plt.savefig(dataset_dir / f"{model_name}_{training_loop}_overlay.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    sns.histplot(scores_forward_np - scores_inverse_np, ax=ax, stat="density")
    ax.set_title(f"{dataset_name} - {model_name} - {training_loop}")
    ax.set_xlabel("Forward - Inverse Score Difference")
    plt.savefig(dataset_dir / f"{model_name}_{training_loop}_residuals.png", dpi=300)
    plt.close(fig)

    return df


if __name__ == "__main__":
    main()
