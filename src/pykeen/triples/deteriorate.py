# -*- coding: utf-8 -*-

"""Deterioration algorithm."""

import math
from typing import List, Union

import torch

from pykeen.triples import TriplesFactory
from pykeen.typing import TorchRandomHint
from pykeen.utils import ensure_torch_random_state

__all__ = [
    'deteriorate',
]


def deteriorate(
    reference: TriplesFactory,
    *others: TriplesFactory,
    n: Union[int, float],
    random_state: TorchRandomHint = None,
) -> List[TriplesFactory]:
    """Remove n triples from the reference set.

    TODO: take care that triples aren't removed that are the only ones with any given entity
    """
    if reference.create_inverse_triples:
        raise NotImplementedError

    if isinstance(n, float):
        if n < 0 or 1 <= n:
            raise ValueError
        n = int(n * reference.num_triples)

    generator = ensure_torch_random_state(random_state)
    idx = torch.randperm(reference.num_triples, generator=generator)
    reference_idx, deteriorated_idx = idx.split(split_size=[reference.num_triples - n, n], dim=0)

    first = reference.clone_and_exchange_triples(
        mapped_triples=reference.mapped_triples[reference_idx],
    )

    # distribute the deteriorated triples across the remaining factories
    didxs = deteriorated_idx.split(math.ceil(n / len(others)), dim=0)
    rest = [
        tf.clone_and_exchange_triples(
            mapped_triples=torch.cat([tf.mapped_triples, reference.mapped_triples[didx]], dim=0),
        )
        for didx, tf in zip(didxs, others)
    ]

    return [first, *rest]


def _main(trials: int = 15):
    from pykeen.datasets import get_dataset
    from pykeen.constants import PYKEEN_EXPERIMENTS
    import numpy as np
    from tabulate import tabulate
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    n_comb = trials * (trials - 1) // 2
    print(f'Number of combinations: {trials} n Choose 2 = {n_comb}')

    rows = []
    for dataset_name in [
        'nations', 'umls', 'kinships',
        # 'codexsmall', 'wn18',
    ]:
        dataset_rows = []
        reference_dataset = get_dataset(dataset=dataset_name)
        for n in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]:
            similarities = []
            for trial in range(trials):
                deteriorated_dataset = reference_dataset.deteriorate(n=n, random_state=trial)
                sim = reference_dataset.similarity(deteriorated_dataset)
                similarities.append(sim)
                rows.append((dataset_name, n, sim))
            dataset_rows.append((n, np.mean(similarities), np.std(similarities)))
        print(tabulate(dataset_rows, headers=[f'{dataset_name} N', 'Mean', 'Std']))

    df = pd.DataFrame(rows, columns=['name', 'n', 'sim'])
    tsv_path = PYKEEN_EXPERIMENTS / 'deteriorating.tsv'
    png_path = PYKEEN_EXPERIMENTS / 'deteriorating.png'
    df.to_csv(tsv_path, sep='\t', index=False)
    sns.lineplot(data=df, x="n", y="sim", hue="name")
    plt.savefig(png_path, dpi=300)


if __name__ == '__main__':
    _main()
