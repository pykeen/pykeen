# -*- coding: utf-8 -*-

"""Remixing and dataset distance utilities.

Most datasets are given in with a pre-defined split, but it's often not discussed
how this split was created. This module contains utilities for investigating the
effects of remixing pre-split datasets like :class`pykeen.datasets.Nations`.

Further, it defines a metric for the "distance" between two splits of a given dataset.
Later, this will be used to map the landscape and see if there is a smooth, continuous
relationship between datasets' splits' distances and their maximum performance.
"""

from typing import List, Sequence

from .splitting import normalize_ratios, split
from .triples_factory import CoreTriplesFactory, cat_triples

__all__ = [
    'remix',
]


def remix(*triples_factories: CoreTriplesFactory, **kwargs) -> List[CoreTriplesFactory]:
    """Remix the triples from the training, testing, and validation set.

    :param triples_factories: A sequence of triples factories
    :param kwargs: Keyword arguments to be passed to :func:`split`
    :returns: A sequence of triples factories of the same sizes but randomly re-assigned triples

    :raises NotImplementedError: if any of the triples factories have ``create_inverse_triples``
    """
    for tf in triples_factories:
        if tf.create_inverse_triples:
            raise NotImplementedError('The remix algorithm is not implemented for datasets with inverse triples')

    all_triples = cat_triples(*triples_factories)
    ratios = _get_ratios(*triples_factories)

    return [
        triples_factories[0].clone_and_exchange_triples(triples)
        for triples in split(all_triples, ratios=ratios, **kwargs)
    ]


def _get_ratios(*triples_factories: CoreTriplesFactory) -> Sequence[float]:
    total = sum(tf.num_triples for tf in triples_factories)
    ratios = normalize_ratios([tf.num_triples / total for tf in triples_factories])
    return ratios


def _main(trials: int = 15):
    from pykeen.datasets import get_dataset
    import numpy as np
    import itertools as itt
    from tqdm import tqdm

    n_comb = trials * (trials - 1) // 2
    print(f'Number of combinations: {trials} n Choose 2 = {n_comb}')

    for dataset_name in [
        'nations', 'umls', 'kinships', 'codexsmall', 'wn18',
    ]:
        reference_dataset = get_dataset(dataset=dataset_name)
        remixed_datasets = [
            reference_dataset.remix(random_state=random_state)
            for random_state in range(trials)
        ]
        similarities = [
            a.similarity(b)
            for a, b in tqdm(
                itt.combinations(remixed_datasets, r=2),
                total=n_comb,
                desc=dataset_name,
            )
        ]
        print(f'[{dataset_name}] Similarities Mean: {np.mean(similarities):.3f}')
        print(f'[{dataset_name}] Similarities Std.: {np.std(similarities):.3f}')
        print(f'[{dataset_name}] Relative Std.: {np.std(similarities) / np.mean(similarities):.3%}')


if __name__ == '__main__':
    _main()
