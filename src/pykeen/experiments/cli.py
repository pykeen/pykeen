# -*- coding: utf-8 -*-

"""Run landmark experiments."""

import logging
import os
import pathlib
import shutil
import sys
import time
from typing import Iterable, Optional, Union
from uuid import uuid4

import click
import tabulate
from more_click import verbose_option
from tqdm.auto import tqdm

from pykeen.utils import CONFIGURATION_FILE_FORMATS, load_configuration, normalize_path

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


@experiments.command()
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

    path = normalize_path(path)

    if not path.is_file():
        click.secho(f"Could not find configuration at {path}", fg="red")
        sys.exit(1)
    click.echo(f"Running configuration at {path}")

    # Create directory in which all experimental artifacts are saved
    datetime = time.strftime("%Y-%m-%d-%H-%M-%S")
    experiment_id = f"{datetime}_{uuid4()}"
    if file_name is not None:
        experiment_id += f"_{file_name}"
    output_directory = normalize_path(directory, experiment_id, mkdir=True)

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


def _iter_configurations() -> Iterable[pathlib.Path]:
    """Iterate over configuration paths."""
    for ext in CONFIGURATION_FILE_FORMATS:
        yield from HERE.rglob(f"*{ext}")


@experiments.command()
def list():
    """List experiment configurations."""
    data = set()
    for path in tqdm(_iter_configurations(), unit="configuration", unit_scale=True, leave=False):
        # clip for node piece configurations
        reference, model, dataset = path.stem.split("_")[:3]
        # "pykeen experiments reproduce" expects "model reference dataset"
        data.add((model, reference, dataset))
    click.secho(f"There are {len(data)} available experiments. Run via\n")
    click.secho("\tpykeen experiments reproduce <MODEL> <REFERENCE> <DATASET>\n")
    click.echo(tabulate.tabulate(sorted(data), headers=("model", "reference", "dataset")))


if __name__ == "__main__":
    experiments()
