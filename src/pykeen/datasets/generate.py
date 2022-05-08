# -*- coding: utf-8 -*-

"""A script for splitting triples into a dataset."""

import json
import pathlib
from typing import Sequence, cast

import click
import numpy as np

from .base import PathDataset
from ..triples import TriplesFactory
from ..utils import normalize_path, random_non_negative_int

LABELS = ["train", "test", "valid"]


@click.command()
@click.argument("path", type=pathlib.Path)
@click.option("-d", "--directory", default=pathlib.Path.cwd(), show_default=True, type=pathlib.Path)
@click.option("--test-ratios", type=float, nargs=2, default=[0.8, 0.2], show_default=True)
@click.option("--no-validation", is_flag=True)
@click.option("--validation-ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1], show_default=True)
@click.option("--reload", is_flag=True)
@click.option("--seed", type=int)
def main(
    path: pathlib.Path,
    directory: pathlib.Path,
    test_ratios: Sequence[float],
    no_validation: bool,
    validation_ratios: Sequence[float],
    reload: bool,
    seed: int,
):
    """Make a dataset from the given triples."""
    directory = normalize_path(directory, mkdir=True)

    # Normalize path
    path = normalize_path(path)
    triples_factory = TriplesFactory.from_path(path=path)
    ratios = test_ratios if no_validation else validation_ratios

    if seed is None:
        seed = random_non_negative_int()
    sub_triples_factories = cast(Sequence[TriplesFactory], triples_factory.split(ratios, random_state=seed))

    for subset_name, subset_tf in zip(LABELS, sub_triples_factories):
        output_path = directory.joinpath(subset_name).with_suffix(".txt")
        click.echo(f"Outputing {subset_name} to {output_path.as_uri()}")
        np.savetxt(output_path, subset_tf.triples, delimiter="\t", fmt="%s")

    metadata = dict(
        source=str(path),
        ratios=dict(zip(LABELS, ratios)),
        seed=seed,
    )
    with directory.joinpath("metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)

    if reload:
        if no_validation:
            click.secho("Can not load as dataset if --no-validation was flagged.", fg="red")
            return
        d = PathDataset(
            training_path=directory.joinpath("train.txt"),
            testing_path=directory.joinpath("test.txt"),
            validation_path=directory.joinpath("valid.txt"),
            eager=True,
        )
        click.echo(d)


if __name__ == "__main__":
    main()
