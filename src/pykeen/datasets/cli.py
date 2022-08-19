# -*- coding: utf-8 -*-

"""Run dataset CLI."""

import itertools as itt
import json
import logging
import math
import pathlib
from textwrap import dedent
from typing import Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union

import click
import docdata
import pandas as pd
import scipy.stats
from more_click import force_option, log_level_option, verbose_option
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .base import Dataset
from .utils import dataset_regex_option, iter_dataset_instances, max_triples_option, min_triples_option
from ..constants import COLUMN_LABELS, PYKEEN_DATASETS
from ..evaluation.evaluator import get_candidate_set_size
from ..metrics.ranking import (
    ArithmeticMeanRank,
    GeometricMeanRank,
    HarmonicMeanRank,
    HitsAtK,
    InverseArithmeticMeanRank,
    InverseGeometricMeanRank,
    InverseHarmonicMeanRank,
    InverseMedianRank,
    MedianRank,
)
from ..triples import CoreTriplesFactory
from ..typing import LABEL_HEAD, LABEL_RELATION, LABEL_TAIL, SIDE_MAPPING, ExtendedTarget

logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()
IMG_DIR = ROOT.joinpath("docs", "source", "img")


@click.group()
def main():
    """Run the dataset CLI."""


@main.command()
@verbose_option
@dataset_regex_option
@min_triples_option
@max_triples_option
def summarize(dataset_regex: Optional[str], min_triples: Optional[int], max_triples: Optional[int]):
    """Load all datasets."""
    for name, dataset in iter_dataset_instances(
        regex_name_filter=dataset_regex, min_triples=min_triples, max_triples=max_triples
    ):
        click.secho(f"Loading {name}", fg="green", bold=True)
        try:
            dataset.summarize(show_examples=None)
        except Exception as e:
            click.secho(f"Failed {name}", fg="red", bold=True)
            click.secho(str(e), fg="red", bold=True)


@main.command()
@verbose_option
@dataset_regex_option
@min_triples_option
@max_triples_option
@force_option
@click.option("--countplots", is_flag=True)
@click.option("-d", "--directory", type=click.Path(dir_okay=True, file_okay=False, resolve_path=True))
def analyze(
    dataset_regex: Optional[str],
    min_triples: Optional[int],
    max_triples: Optional[int],
    force: bool,
    countplots: bool,
    directory,
):
    """Generate analysis."""
    for name, dataset in iter_dataset_instances(
        regex_name_filter=dataset_regex, min_triples=min_triples, max_triples=max_triples
    ):
        _analyze(
            dataset_name=name,
            dataset=dataset,
            force=force,
            countplots=countplots,
            directory=directory,
        )


def _analyze(
    dataset_name: str,
    dataset: Dataset,
    force: bool,
    countplots: bool,
    directory: Union[None, str, pathlib.Path],
):
    from . import analysis

    plt, sns = _get_plotting_libraries()

    # Raise matplotlib level
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if directory is None:
        directory = PYKEEN_DATASETS
    else:
        directory = pathlib.Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

    d = directory.joinpath(dataset_name, "analysis")
    d.mkdir(parents=True, exist_ok=True)

    dfs = {}
    it = tqdm(analysis.__dict__.items(), leave=False, desc="Stats")
    for name, func in it:
        if not name.startswith("get") or not name.endswith("df"):
            continue
        it.set_postfix(func=name)
        key = name[len("get_") : -len("_df")]
        path = d.joinpath(key).with_suffix(".tsv")
        if path.exists() and not force:
            df = pd.read_csv(path, sep="\t")
        else:
            df = func(dataset=dataset)
            df.to_csv(d.joinpath(key).with_suffix(".tsv"), sep="\t", index=False)
        dfs[key] = df

    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(
        data=dfs["relation_injectivity"],
        x=LABEL_HEAD,
        y=LABEL_TAIL,
        size="support",
        hue="support",
        ax=ax,
    )
    ax.set_title(f'{docdata.get_docdata(dataset.__class__)["name"]} Relation Injectivity')
    fig.tight_layout()
    fig.savefig(d.joinpath("relation_injectivity.svg"))
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(
        data=dfs["relation_functionality"],
        x="functionality",
        y="inverse_functionality",
        ax=ax,
    )
    ax.set_title(f'{docdata.get_docdata(dataset.__class__)["name"]} Relation Functionality')
    fig.tight_layout()
    fig.savefig(d.joinpath("relation_functionality.svg"))
    plt.close(fig)

    if countplots:
        entity_count_df = (
            dfs["entity_count"].groupby("entity_label").sum().reset_index().sort_values("count", ascending=False)
        )
        fig, ax = plt.subplots(1, 1)
        sns.barplot(data=entity_count_df, y="entity_label", x="count", ax=ax)
        ax.set_ylabel("")
        ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(d.joinpath("entity_counts.svg"))
        plt.close(fig)

        relation_count_df = (
            dfs["relation_count"].groupby("relation_label").sum().reset_index().sort_values("count", ascending=False)
        )
        fig, ax = plt.subplots(1, 1)
        sns.barplot(data=relation_count_df, y="relation_label", x="count", ax=ax)
        ax.set_ylabel("")
        ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(d.joinpath("relation_counts.svg"))
        plt.close(fig)


def _get_plotting_libraries():
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError(
            dedent(
                """\
            Please install plotting dependencies by

                pip install pykeen[plotting]

            or directly by

                pip install matplotlib seaborn
        """
            )
        )
    return plt, sns


@main.command()
@verbose_option
@dataset_regex_option
@min_triples_option
@max_triples_option
def verify(dataset_regex: Optional[str], min_triples: Optional[int], max_triples: Optional[int]):
    """Verify dataset integrity."""
    data = []
    keys = None
    for name, dataset_instance in iter_dataset_instances(
        regex_name_filter=dataset_regex, min_triples=min_triples, max_triples=max_triples
    ):
        data.append(
            list(
                itt.chain(
                    [name],
                    itt.chain.from_iterable(
                        (triples_factory.num_entities, triples_factory.num_relations)
                        for _, triples_factory in sorted(dataset_instance.factory_dict.items())
                    ),
                )
            )
        )
        keys = keys or sorted(dataset_instance.factory_dict.keys())
    if not keys:
        return
    df = pd.DataFrame(
        data=data,
        columns=["name"] + [f"num_{part}_{a}" for part in keys for a in ("entities", "relations")],
    )
    valid = None
    for part, a in itt.product(("validation", "testing"), ("entities", "relations")):
        this_valid = df[f"num_training_{a}"] == df[f"num_{part}_{a}"]
        if valid is None:
            valid = this_valid
        else:
            valid = valid & this_valid
    df["valid"] = valid
    click.echo(df.to_markdown())


@main.command()
@verbose_option
@dataset_regex_option
@max_triples_option
@min_triples_option
@click.option(
    "-s",
    "--samples",
    type=int,
    default=10_000,
    show_default=True,
    help="Number of samples for estimating expected values",
)
@log_level_option(default=logging.ERROR)
@force_option
@click.option("--output-directory", default=PYKEEN_DATASETS, type=pathlib.Path, show_default=True)
def expected_metrics(
    dataset_regex: Optional[str],
    max_triples: Optional[int],
    min_triples: Optional[int],
    log_level: str,
    samples: int,
    force: bool,
    output_directory: pathlib.Path,
):
    """Compute expected metrics for all datasets (matching the given pattern)."""
    logging.getLogger("pykeen").setLevel(level=log_level)
    df_data: List[Tuple[str, str, str, str, float]] = []
    for dataset_name, dataset_instance in iter_dataset_instances(
        regex_name_filter=dataset_regex, max_triples=max_triples, min_triples=min_triples
    ):
        adjustments_directory = output_directory.joinpath(dataset_name, "adjustments")
        adjustments_directory.mkdir(parents=True, exist_ok=True)
        expected_metrics_path = adjustments_directory.joinpath("expected_metrics.json")
        if expected_metrics_path.is_file() and not force:
            expected_metrics_dict = json.loads(expected_metrics_path.read_text())
        else:
            expected_metrics_dict = dict()
            for key, factory in dataset_instance.factory_dict.items():
                if key == "training":
                    additional_filter_triples = None
                elif key == "validation":
                    additional_filter_triples = dataset_instance.training.mapped_triples
                elif key == "testing":
                    additional_filter_triples = [
                        dataset_instance.training.mapped_triples,
                    ]
                    if dataset_instance.validation is None:
                        click.echo(f"WARNING: {dataset_name} does not have validation triples!")
                    else:
                        additional_filter_triples.append(dataset_instance.validation.mapped_triples)
                else:
                    raise AssertionError(key)
                df = get_candidate_set_size(
                    mapped_triples=factory.mapped_triples,
                    additional_filter_triples=additional_filter_triples,
                )
                output_path = adjustments_directory.joinpath(f"{key}_candidates.tsv.gz")
                df.to_csv(output_path, sep="\t", index=False)
                tqdm.write(f"wrote {output_path}")

                # expected metrics
                ks = (1, 3, 5, 10) + tuple(
                    10**i for i in range(2, int(math.ceil(math.log(dataset_instance.num_entities))))
                )
                metrics = [
                    ArithmeticMeanRank(),
                    *(HitsAtK(k) for k in ks),
                    InverseHarmonicMeanRank(),
                    # Needs simulation
                    InverseArithmeticMeanRank(),
                    HarmonicMeanRank(),
                    GeometricMeanRank(),
                    InverseGeometricMeanRank(),
                    MedianRank(),
                    InverseMedianRank(),
                ]
                this_metrics: MutableMapping[ExtendedTarget, Mapping[str, float]] = dict()
                for label, sides in SIDE_MAPPING.items():
                    num_candidates = df[[f"{side}_candidates" for side in sides]].values.ravel()
                    this_metrics[label] = {
                        metric.key: metric.expected_value(
                            num_candidates=num_candidates,
                            num_samples=samples,
                        )
                        for metric in metrics
                    }
                expected_metrics_dict[key] = this_metrics
            with expected_metrics_path.open("w") as file:
                json.dump(expected_metrics_dict, file, sort_keys=True, indent=4)
            tqdm.write(f"wrote {expected_metrics_path}")

        df_data.extend(
            (dataset_name, metric, side, part, value)
            for part, level1 in expected_metrics_dict.items()
            for side, level2 in level1.items()
            for metric, value in level2.items()
        )
    df = (
        pd.DataFrame(df_data, columns=["dataset", "metric", "side", "split", "value"])
        .sort_values(
            by=["dataset", "metric", "side", "split"],
        )
        .reset_index(drop=True)
    )
    results_path = output_directory.joinpath("metric_adjustments.tsv.gz")
    df.to_csv(results_path, sep="\t", index=False)
    click.secho(f"wrote to {results_path}", fg="green")

    if max_triples is None and min_triples is None and dataset_regex is None:
        try:
            from zenodo_client import update_zenodo
        except ImportError:
            click.secho("Unable to import `zenodo_client`. Not uploading results", fg="red")
        else:
            zenodo_record = "6331629"
            # See https://zenodo.org/record/6331629
            rv = update_zenodo(zenodo_record, results_path)
            click.secho(f"Updated Zenodo record {zenodo_record}: {rv}", fg="green")


def _summarize_degree_distribution(factory: CoreTriplesFactory) -> Iterable[List]:
    df = pd.DataFrame(data=factory.mapped_triples.numpy(), columns=COLUMN_LABELS)
    for target in [LABEL_HEAD, LABEL_TAIL]:
        key = [LABEL_TAIL if target == LABEL_HEAD else LABEL_HEAD, LABEL_RELATION]
        unique_targets = df.groupby(by=key)[target].nunique()
        yield [target, *scipy.stats.describe(unique_targets)]


# TODO: maybe merge into analyze / make sub-command
@main.command()
@verbose_option
@dataset_regex_option
@min_triples_option
@max_triples_option
@click.option(
    "-r",
    "--restrict-split",
    type=click.Choice(["testing", "training", "validation"], case_sensitive=False),
    default=None,
)
@force_option
@click.option("--plot", is_flag=True)
@click.option("-o", "--output-root", type=pathlib.Path, default=PYKEEN_DATASETS.joinpath("analysis"))
def degree(
    dataset_regex: Optional[str],
    min_triples: Optional[int],
    max_triples: Optional[int],
    restrict_split: Optional[str],
    force: bool,
    plot: bool,
    output_root: pathlib.Path,
):
    """Analyze degree distributions."""
    output_root.mkdir(exist_ok=True, parents=True)
    base_path = output_root.joinpath("degree-distributions")
    path = base_path.with_suffix(suffix=".tsv.gz")
    if path.is_file() and not force:
        df = pd.read_csv(path, sep="\t")
        logger.info(f"Loaded degree statistics from {path}")
    else:
        with logging_redirect_tqdm():
            rows = [
                (name, split, factory.num_triples, *row)
                for name, dataset in iter_dataset_instances(
                    regex_name_filter=dataset_regex, min_triples=min_triples, max_triples=max_triples
                )
                for split, factory in dataset.factory_dict.items()
                if (restrict_split is None or split == restrict_split)
                for row in _summarize_degree_distribution(factory=factory)
            ]
        df = pd.DataFrame(
            data=rows,
            columns=[
                "dataset",
                "split",
                "num_triples",
                "target",
                "nobs",
                "minmax",
                "mean",
                "variance",
                "skewness",
                "kurtosis",
            ],
        )
        # only save full data
        if dataset_regex is None and min_triples is None and max_triples is None and restrict_split is None:
            df.to_csv(path, sep="\t", index=False)
            logger.info(f"Written degree statistics to {path}")
    if not plot:
        return
    plt, sns = _get_plotting_libraries()

    # Plot: Descriptive Statistics of Degree Distributions per dataset / split vs. number of triples (=size)
    df = df.melt(
        id_vars=["dataset", "split", "num_triples", "target"],
        value_vars=["mean", "variance", "skewness", "kurtosis"],
        var_name="statistic",
    )
    grid_1: sns.FacetGrid = sns.relplot(  # type: ignore
        data=df,
        hue="dataset",
        x="num_triples",
        style=None if restrict_split is not None else "split",
        col="statistic",
        row="target",
        y="value",
        facet_kws=dict(
            margin_titles=True,
            sharey="col",
        ),
        height=2.5,
        hue_order=sorted(df["dataset"].unique()),
    )
    grid_1.fig.suptitle("Dataset Degree Distributions", x=0.4, y=0.98)
    plt.subplots_adjust(top=0.85)
    sns.move_legend(
        grid_1,
        "lower center",
        bbox_to_anchor=(0.45, -0.35),
        ncol=6,
        title=None,
        frameon=False,
    )
    grid_1.tight_layout()
    grid_1.set(xscale="log", yscale="log", xlabel="Triples")
    path = base_path.with_suffix(suffix=".pdf")
    grid_1.savefig(path)
    grid_1.savefig(IMG_DIR.joinpath("dataset_degree_distributions.svg"))
    logger.info(f"Saved plot to {path}")

    # Plot: difference between mean head and tail degree
    df_2 = df.loc[df["statistic"] == "mean"].pivot(
        index=["dataset", "split", "num_triples"], columns="target", values="value"
    )
    df_2["difference"] = df_2["head"] - df_2["tail"]
    grid_2: sns.FacetGrid = sns.relplot(  # type: ignore
        data=df_2,
        hue="dataset",
        x="num_triples",
        style=None if restrict_split is not None else "split",
        y="difference",
        height=2.5,
        aspect=4,
        hue_order=sorted(df["dataset"].unique()),
    )
    grid_2.fig.suptitle("Dataset Mean Degree Imbalance", x=0.4, y=0.98)
    sns.move_legend(
        grid_2,
        "lower center",
        bbox_to_anchor=(0.45, -0.55),
        ncol=6,
        title=None,
        frameon=False,
    )
    grid_2.tight_layout()
    grid_2.set(xscale="log", yscale="symlog", xlabel="Triples")
    path = base_path.with_name("degree-imbalance").with_suffix(suffix=".pdf")
    grid_2.savefig(path)
    grid_2.savefig(IMG_DIR.joinpath("degree_imbalance.svg"))
    logger.info(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
