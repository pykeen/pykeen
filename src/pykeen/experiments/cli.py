# -*- coding: utf-8 -*-

"""Run landmark experiments."""

import logging
import os
import pathlib
import shutil
import sys
import time
from collections import defaultdict
from typing import Any, Iterable, List, Mapping, Optional, Tuple, Type, Union
from uuid import uuid4

import click
import numpy
import pandas
import tabulate
from class_resolver import get_subclasses
from more_click import verbose_option

from pykeen.datasets import get_dataset
from pykeen.evaluation.evaluator import get_candidate_set_size
from pykeen.evaluation.ranking_metric_lookup import MetricKey, normalize_flattened_metric_results
from pykeen.metrics import RankBasedMetric, rank_based_metric_resolver
from pykeen.metrics.ranking import DerivedRankBasedMetric, HitsAtK
from pykeen.utils import CONFIGURATION_FILE_FORMATS, load_configuration

__all__ = [
    "experiments",
]

logger = logging.getLogger(__name__)
HERE = pathlib.Path(__file__).parent.resolve()


def _make_dir(_ctx, _param, value):
    os.makedirs(value, exist_ok=True)
    return value


directory_option = click.option(
    "-d",
    "--directory",
    type=click.Path(dir_okay=True, file_okay=False),
    callback=_make_dir,
    default=os.getcwd(),
)
replicates_option = click.option(
    "-r",
    "--replicates",
    type=int,
    default=1,
    show_default=True,
    help="Number of times to retrain the model.",
)
move_to_cpu_option = click.option("--move-to-cpu", is_flag=True, help="Move trained model(s) to CPU after training.")
discard_replicates_option = click.option(
    "--discard-replicates",
    is_flag=True,
    help="Discard trained models after training.",
)
extra_config_option = click.option(
    "--extra-config",
    type=pathlib.Path,
    default=None,
    help="Path to a file with additional configuration, e.g., to add a result tracker.",
)
keep_seed_option = click.option(
    "--keep-seed",
    is_flag=True,
    help=(
        "If a random seed is given in the configuration, keep it rather than discarding. "
        "Notice that this will render multiple replicates useless."
    ),
)


@click.group()
def experiments():
    """Run landmark experiments."""


@experiments.command(
    epilog="Available experiments:\n\n\b\n"
    + tabulate.tabulate(
        sorted(path.stem.split("_") for ext in CONFIGURATION_FILE_FORMATS for path in HERE.rglob(f"*{ext}")),
        headers=("reference", "model", "dataset"),
    ),
)
@click.argument("model")
@click.argument("reference")
@click.argument("dataset")
@replicates_option
@move_to_cpu_option
@discard_replicates_option
@directory_option
@verbose_option
@extra_config_option
@keep_seed_option
def reproduce(
    model: str,
    reference: str,
    dataset: str,
    replicates: int,
    directory: str,
    move_to_cpu: bool,
    discard_replicates: bool,
    extra_config: Optional[pathlib.Path],
    keep_seed: bool,
):
    """Reproduce a pre-defined experiment included in PyKEEN.

    Example: $ pykeen experiments reproduce tucker balazevic2019 fb15k
    """
    file_name = f"{reference}_{model}_{dataset}"
    path = HERE.joinpath(model, file_name)
    paths = {full_path for full_path in map(path.with_suffix, CONFIGURATION_FILE_FORMATS) if full_path.is_file()}
    if len(paths) == 0:
        raise FileNotFoundError("Could not find a configuration file.")
    elif len(paths) > 1:
        raise ValueError(f"Found multiple configuration files: {paths}")
    path = next(iter(paths))
    _help_reproduce(
        directory=directory,
        path=path,
        replicates=replicates,
        move_to_cpu=move_to_cpu,
        save_replicates=not discard_replicates,
        file_name=file_name,
        extra_config=extra_config,
        keep_seed=keep_seed,
    )


@experiments.command()
@click.argument("path")
@replicates_option
@move_to_cpu_option
@discard_replicates_option
@directory_option
@extra_config_option
@keep_seed_option
@verbose_option
def run(
    path: str,
    replicates: int,
    directory: str,
    move_to_cpu: bool,
    discard_replicates: bool,
    extra_config: Optional[pathlib.Path],
    keep_seed: bool,
):
    """Run a single reproduction experiment."""
    _help_reproduce(
        path=path,
        replicates=replicates,
        directory=directory,
        move_to_cpu=move_to_cpu,
        save_replicates=not discard_replicates,
        extra_config=extra_config,
        keep_seed=keep_seed,
    )


def _help_reproduce(
    *,
    directory: Union[str, pathlib.Path],
    path: Union[str, pathlib.Path],
    replicates: int,
    move_to_cpu: bool = False,
    save_replicates: bool = True,
    file_name: Optional[str] = None,
    extra_config: Optional[pathlib.Path] = None,
    keep_seed: bool = False,
) -> None:
    """Help run the configuration at a given path.

    :param directory: Output directory
    :param path: Path to configuration JSON/YAML file
    :param replicates: How many times the experiment should be run
    :param move_to_cpu: Should the model be moved back to the CPU? Only relevant if training on GPU.
    :param save_replicates: Should the artifacts of the replicates be saved?
    :param file_name: Name of JSON/YAML file (optional)
    :param keep_seed:
        whether to keep a random seed if given as part of the configuration
    """
    from pykeen.pipeline import replicate_pipeline_from_path

    if isinstance(path, str):
        path = pathlib.Path(path).resolve()

    if not path.is_file():
        click.secho(f"Could not find configuration at {path}", fg="red")
        sys.exit(1)
    click.echo(f"Running configuration at {path}")

    # Create directory in which all experimental artifacts are saved
    datetime = time.strftime("%Y-%m-%d-%H-%M-%S")
    if file_name is not None:
        experiment_id = f"{datetime}_{uuid4()}_{file_name}"
    else:
        experiment_id = f"{datetime}_{uuid4()}"

    if isinstance(directory, str):
        directory = pathlib.Path(directory).resolve()
    output_directory = directory.joinpath(experiment_id)
    output_directory.mkdir(exist_ok=True, parents=True)

    extra_kwargs = {} if extra_config is None else load_configuration(path=extra_config)

    replicate_pipeline_from_path(
        path=path,
        directory=output_directory,
        replicates=replicates,
        use_testing_data=True,
        move_to_cpu=move_to_cpu,
        save_replicates=save_replicates,
        keep_seed=keep_seed,
        **extra_kwargs,
    )
    shutil.copyfile(path, output_directory.joinpath("configuration_copied").with_suffix(path.suffix))


@experiments.command()
@click.argument("path")
@verbose_option
@click.option("-d", "--directory", type=click.Path(file_okay=False, dir_okay=True))
def optimize(path: str, directory: str):
    """Run a single HPO experiment."""
    from pykeen.hpo import hpo_pipeline_from_path

    hpo_pipeline_result = hpo_pipeline_from_path(path)
    hpo_pipeline_result.save_to_directory(directory)


@experiments.command()
@click.argument("path", type=click.Path(file_okay=True, dir_okay=False, exists=True))
@directory_option
@click.option("--dry-run", is_flag=True)
@click.option("-r", "--best-replicates", type=int, help="Number of times to retrain the best model.")
@move_to_cpu_option
@discard_replicates_option
@click.option("-s", "--save-artifacts", is_flag=True)
@verbose_option
def ablation(
    path: str,
    directory: str,
    dry_run: bool,
    best_replicates: int,
    save_artifacts: bool,
    move_to_cpu: bool,
    discard_replicates: bool,
) -> None:
    """Generate a set of HPO configurations.

    A sample file can be run with ``pykeen experiments ablation tests/resources/hpo_complex_nations.json``.
    """
    from ..ablation.ablation import _run_ablation_experiments, prepare_ablation_from_path

    directories = prepare_ablation_from_path(path=path, directory=directory, save_artifacts=save_artifacts)

    _run_ablation_experiments(
        directories=directories,
        best_replicates=best_replicates,
        dry_run=dry_run,
        move_to_cpu=move_to_cpu,
        discard_replicates=discard_replicates,
    )


@experiments.command()
def validate():
    """Validate configurations."""
    from .validate import get_configuration_errors, iterate_config_paths

    has_error = False
    for _directory_name, _config_name, path in iterate_config_paths():
        path = path.resolve()
        errors = get_configuration_errors(path=path)
        if errors:
            click.secho(f"Errors in {path.as_uri()}")
        for error in errors:
            click.secho(error, err=True, color=True)
            has_error = True
    exit(-1 if has_error else 0)


def _iter_results(configuration: Mapping[str, Any]) -> Iterable[Tuple[RankBasedMetric, float]]:
    for metric, value in normalize_flattened_metric_results(configuration.get("results", {})).items():
        key = MetricKey.lookup(metric)
        kwargs = {}
        metric_name = key.metric
        if metric_name.startswith("hits_at_"):
            kwargs["k"] = int(metric_name[len("hits_at_") :])
            metric_name = "hits_at_k"
        metric_instance = rank_based_metric_resolver.make(metric_name, pos_kwargs=kwargs)
        yield metric_instance, value


@experiments.command(name="post-adjust")
def post_adjust():
    """Calculate adjusted metrics from published raw metrics without access to the model."""
    from .validate import iterate_config_paths

    # index adjusted metrics
    index: Mapping[Type[RankBasedMetric], List[Type[DerivedRankBasedMetric]]] = defaultdict(list)
    for cls in get_subclasses(cls=DerivedRankBasedMetric):
        base = cls.base_cls
        if base is not None:
            index[base].append(cls)

    data = []
    for _directory_name, _config_name, path in iterate_config_paths():
        config = load_configuration(path)
        if not config:
            logger.error(f"Invalid configuration at {path}")
            continue
        dataset = get_dataset(dataset=config.get("pipeline", {}).get("dataset", None))
        _kwargs = config.get("pipeline", {}).get("evaluator_kwargs", {}) or {}
        if _kwargs.get("filtered", True):
            additional_filter_triples = [
                dataset.training.mapped_triples,
                dataset.validation.mapped_triples,
            ]
        else:
            raise ValueError
        css = get_candidate_set_size(
            mapped_triples=dataset.testing.mapped_triples,
            additional_filter_triples=additional_filter_triples,
        )
        num_candidates = numpy.concatenate([css["head_candidates"].values, css["tail_candidates"].values])
        model = path.parent.name
        for metric, value in _iter_results(configuration=config):
            adjustments = index[type(metric)]
            if not adjustments:
                continue
            kwargs = dict(k=metric.k) if isinstance(metric, HitsAtK) else {}
            data.append([model, dataset.get_normalized_name(), metric.key, value, path.name])
            for adjusted_metric_cls in adjustments:
                adjusted_metric = rank_based_metric_resolver.make(adjusted_metric_cls, pos_kwargs=kwargs)
                assert isinstance(adjusted_metric, DerivedRankBasedMetric)
                adjusted_value = adjusted_metric.adjust(base_metric_result=value, num_candidates=num_candidates)
                data.append([model, dataset.get_normalized_name(), adjusted_metric.key, adjusted_value, path.name])
    df = pandas.DataFrame(data=data, columns=["model", "dataset", "metric", "value", "path"])
    df.to_csv("/tmp/post_adjustments.tsv", sep="\t", index=False)


if __name__ == "__main__":
    experiments()
