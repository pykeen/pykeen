# -*- coding: utf-8 -*-

"""Deterioration algorithm."""

import logging
import math
from itertools import zip_longest
from typing import List, Union

import click
import more_click
import torch

from pykeen.triples import TriplesFactory
from pykeen.typing import TorchRandomHint
from pykeen.utils import ensure_torch_random_state

__all__ = [
    "deteriorate",
]

logger = logging.getLogger(__name__)


def deteriorate(
    reference: TriplesFactory,
    *others: TriplesFactory,
    n: Union[int, float],
    random_state: TorchRandomHint = None,
) -> List[TriplesFactory]:
    """Remove n triples from the reference set.

    :param reference: The reference triples factory
    :param others: Other triples factories to deteriorate
    :param n: The ratio to deteriorate. If given as a float, should be between 0 and 1.
        If an integer, deteriorates that many triples
    :param random_state: The random state
    :returns: A concatenated list of the processed reference and other triples factories
    :raises NotImplementedError: if the reference triples factory has inverse triples
    :raises ValueError: If a float is given for n that isn't between 0 and 1
    """
    # TODO: take care that triples aren't removed that are the only ones with any given entity
    if reference.create_inverse_triples:
        raise NotImplementedError

    if isinstance(n, float):
        if n < 0 or 1 <= n:
            raise ValueError
        n = int(n * reference.num_triples)

    generator = ensure_torch_random_state(random_state)
    logger.debug("random state %s", random_state)
    logger.debug("generator %s %s", generator, generator.get_state())
    idx = torch.randperm(reference.num_triples, generator=generator)
    logger.debug("idx %s", idx)
    reference_idx, deteriorated_idx = idx.split(split_size=[reference.num_triples - n, n], dim=0)

    first = reference.clone_and_exchange_triples(
        mapped_triples=reference.mapped_triples[reference_idx],
    )

    # distribute the deteriorated triples across the remaining factories
    didxs = deteriorated_idx.split(math.ceil(n / len(others)), dim=0)
    rest = [
        tf.clone_and_exchange_triples(
            mapped_triples=(
                torch.cat([tf.mapped_triples, reference.mapped_triples[didx]], dim=0)
                if didx is not None
                else tf.mapped_triples  # maybe just give same tf? should it be copied?
            ),
        )
        for didx, tf in zip_longest(didxs, others)
    ]

    return [first, *rest]


@click.command()
@more_click.verbose_option
@click.option("--trials", type=int, default=15, show_default=True)
def _main(trials: int):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from tabulate import tabulate

    from pykeen.constants import PYKEEN_EXPERIMENTS
    from pykeen.datasets import get_dataset

    n_comb = trials * (trials - 1) // 2
    logger.info(f"Number of combinations: {trials} n Choose 2 = {n_comb}")

    ns = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    rows = []
    for dataset_name in [
        # 'kinships',
        "nations",
        # 'umls',
        # 'codexsmall',
        # 'wn18',
    ]:
        dataset_rows = []
        reference_dataset = get_dataset(dataset=dataset_name)
        for n in ns:
            similarities = []
            for trial in range(trials):
                deteriorated_dataset = reference_dataset.deteriorate(n=n, random_state=trial)
                sim = reference_dataset.similarity(deteriorated_dataset)
                similarities.append(sim)
                rows.append((dataset_name, n, trial, sim))
            dataset_rows.append((n, np.mean(similarities), np.std(similarities)))
        click.echo(tabulate(dataset_rows, headers=[f"{dataset_name} N", "Mean", "Std"]))

    df = pd.DataFrame(rows, columns=["name", "n", "trial", "sim"])
    tsv_path = PYKEEN_EXPERIMENTS / "deteriorating.tsv"
    png_path = PYKEEN_EXPERIMENTS / "deteriorating.png"
    click.echo(f"writing to {tsv_path}")
    df.to_csv(tsv_path, sep="\t", index=False)
    sns.lineplot(data=df, x="n", y="sim", hue="name")
    plt.savefig(png_path, dpi=300)


if __name__ == "__main__":
    _main()
