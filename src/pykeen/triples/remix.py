# -*- coding: utf-8 -*-

"""Remixing and dataset distance utilities.

Most datasets are given in with a pre-defined split, but it's often not discussed
how this split was created. This module contains utilities for investigating the
effects of remixing pre-split datasets like :class`pykeen.datasets.Nations`.

Further, it defines a metric for the "distance" between two splits of a given dataset.
Later, this will be used to map the landscape and see if there is a smooth, continuous
relationship between datasets' splits' distances and their maximum performance.
"""

from typing import Sequence

from .triples_factory import TriplesFactory
from .utils import calculate_ratios, concatenate_triples_factories, summarize
from ..datasets.base import DataSet, EagerDataset

__all__ = [
    'remix_dataset',
    'remix',
    'deteriorate',
    'dataset_splits_distance',
    'splits_distance',
]


def remix_dataset(dataset: DataSet, **kwargs) -> DataSet:
    """Remix a dataset."""
    return EagerDataset(*remix(dataset.training, dataset.testing, dataset.validation, **kwargs))


def remix(*triples_factories: TriplesFactory, **kwargs):
    """Remix the triples from the training, testing, and validation set."""
    triples = concatenate_triples_factories(*triples_factories)
    ratios = calculate_ratios(*triples_factories)

    tf = TriplesFactory(
        triples=triples,
        entity_to_id=triples_factories[0].entity_to_id,
        relation_to_id=triples_factories[0].relation_to_id,
        # FIXME doesn't carry flag of create_inverse_triples through
    )
    return tf.split(ratios=ratios, **kwargs)


def deteriorate(reference: TriplesFactory, *triples_factories: TriplesFactory, n: int):
    """Remove n triples from the reference set.

    Take care that triples aren't removed that are the only ones with any given entity
    """
    raise NotImplementedError


def dataset_splits_distance(a: DataSet, b: DataSet) -> float:
    """Compute the distance between two datasets that are remixes of each other via :func:`splits_distance`."""
    return splits_distance(
        (a.training, a.testing, a.validation),
        (b.training, b.testing, b.validation),
    )


def splits_distance(a: Sequence[TriplesFactory], b: Sequence[TriplesFactory]) -> float:
    """Compute the distance between two datasets' splits.

    :return: The number of triples present in the training sets in both
    """
    if len(a) != len(b):
        raise ValueError('Must have same number')

    # concatenate test and valid
    train_1 = _smt(a[0].triples)
    train_2 = _smt(b[0].triples)
    non_train_1 = _smt(concatenate_triples_factories(*a[1:]))
    # non_train_2 = smt(concatenate_triples_factories(test_2, valid_2))
    # TODO more interesting way to discuss splits w/ valid
    return len(train_1.intersection(train_2)) / len(train_1.union(non_train_1))


def _smt(x):
    return set(tuple(xx) for xx in x)


def _main():
    from pykeen.datasets import Nations
    import numpy as np
    import random
    import itertools as itt
    n = Nations()
    summarize(n.training, n.testing, n.validation)

    trials = 35
    splits = [
        remix(
            n.training, n.testing, n.validation,
            random_state=random.randint(2, 2 * 25),
        )
        for _ in range(trials)
    ]
    distances = [
        splits_distance(a, b)
        for a, b in itt.combinations(splits, r=2)
    ]

    print(f'Distances number (1/2) * {trials} * ({trials}-1) = {len(distances)}')
    print('Distances Mean', np.mean(distances))
    print('Distances Std.', np.std(distances))


if __name__ == '__main__':
    _main()
