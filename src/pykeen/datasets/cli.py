# -*- coding: utf-8 -*-

"""Run dataset CLI."""

import itertools as itt
import json
import logging
import math
import pathlib
from textwrap import dedent
from typing import Iterable, List, Mapping, MutableMapping, Optional, Tuple, Type, Union

import click
import docdata
import pandas as pd
from more_click import force_option, log_level_option, verbose_option
from tqdm import tqdm

from . import dataset_resolver, get_dataset
from ..constants import PYKEEN_DATASETS
from ..datasets.base import Dataset
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
from ..typing import LABEL_HEAD, LABEL_TAIL, SIDE_MAPPING, ExtendedTarget


@click.group()
def main():
    """Run the dataset CLI."""


@main.command()
@verbose_option
def summarize():
    """Load all datasets."""
    for name, dataset in _iter_datasets():
        click.secho(f"Loading {name}", fg="green", bold=True)
        try:
            dataset().summarize(show_examples=None)
        except Exception as e:
            click.secho(f"Failed {name}", fg="red", bold=True)
            click.secho(str(e), fg="red", bold=True)


def _get_num_triples(pair: Tuple[str, Type[Dataset]]) -> int:
    """Extract the number of triples from docdata."""
    return docdata.get_docdata(pair[1])["statistics"]["triples"]


def _iter_datasets(
    regex_name_filter=None, *, max_triples: Optional[int] = None, min_triples: Optional[int] = None
) -> Iterable[Tuple[str, Type[Dataset]]]:
    it = sorted(
        dataset_resolver.lookup_dict.items(),
        key=_get_num_triples,
    )
    if max_triples is not None:
        it = [pair for pair in it if _get_num_triples(pair) <= max_triples]
    if min_triples is not None:
        it = [pair for pair in it if _get_num_triples(pair) >= min_triples]
    if regex_name_filter is not None:
        if isinstance(regex_name_filter, str):
            import re

            regex_name_filter = re.compile(regex_name_filter)
        it = [(name, dataset) for name, dataset in it if regex_name_filter.match(name)]
    it_tqdm = tqdm(
        it,
        desc="Datasets",
    )
    for k, v in it_tqdm:
        n_triples = docdata.get_docdata(v)["statistics"]["triples"]
        it_tqdm.set_postfix(name=k, triples=f"{n_triples:,}")
        yield k, v


@main.command()
@verbose_option
@click.option("--dataset", help="Regex for filtering datasets by name")
@click.option("-f", "--force", is_flag=True)
@click.option("--countplots", is_flag=True)
@click.option("-d", "--directory", type=click.Path(dir_okay=True, file_okay=False, resolve_path=True))
def analyze(dataset, force: bool, countplots: bool, directory):
    """Generate analysis."""
    for _name, dataset in _iter_datasets(regex_name_filter=dataset):
        _analyze(dataset, force, countplots, directory=directory)


def _analyze(dataset, force, countplots, directory: Union[None, str, pathlib.Path]):
    from . import analysis

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

    # Raise matplotlib level
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if directory is None:
        directory = PYKEEN_DATASETS
    else:
        directory = pathlib.Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

    dataset_instance = get_dataset(dataset=dataset)
    dataset_name = dataset_instance.get_normalized_name()
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
            df = func(dataset=dataset_instance)
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
    ax.set_title(f'{docdata.get_docdata(dataset_instance.__class__)["name"]} Relation Injectivity')
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
    ax.set_title(f'{docdata.get_docdata(dataset_instance.__class__)["name"]} Relation Functionality')
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


@main.command()
@verbose_option
@click.option("--dataset", help="Regex for filtering datasets by name")
def verify(dataset: str):
    """Verify dataset integrity."""
    data = []
    keys = None
    for name, dataset_cls in _iter_datasets(regex_name_filter=dataset):
        dataset_instance = get_dataset(dataset=dataset_cls)
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
@click.option("-d", "--dataset", help="Regex for filtering datasets by name")
@click.option("-m", "--max-triples", type=int, default=None)
@click.option("--min-triples", type=int, default=None)
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
    dataset: Optional[str],
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
    for _dataset_name, dataset_cls in _iter_datasets(
        regex_name_filter=dataset, max_triples=max_triples, min_triples=min_triples
    ):
        dataset_instance = get_dataset(dataset=dataset_cls)
        dataset_name = dataset_resolver.normalize_inst(dataset_instance)
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
        pd.DataFrame(df_data, columns=["dataset", "metric", "side", "part", "value"])
        .sort_values(
            by=["dataset", "metric", "side", "part"],
        )
        .reset_index(drop=True)
    )
    results_path = output_directory.joinpath("metric_adjustments.tsv.gz")
    df.to_csv(results_path, sep="\t", index=False)
    click.secho(f"wrote {results_path}")
    click.echo(df.to_markdown(index=False))

    if max_triples is None and min_triples is None and dataset is None:
        try:
            from zenodo_client import update_zenodo
        except ImportError:
            return
        else:
            zenodo_record = "6331629"
            # See https://zenodo.org/record/6331629
            rv = update_zenodo(zenodo_record, results_path)
            click.secho(f"Updated Zenodo record {zenodo_record}: {rv}", fg="green")


if __name__ == "__main__":
    main()
