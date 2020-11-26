# -*- coding: utf-8 -*-

"""Remixing and dataset distance utilities.

Most datasets are given in with a pre-defined split, but it's often not discussed
how this split was created. This module contains utilities for investigating the
effects of remixing pre-split datasets like :class`pykeen.datasets.Nations`.

Further, it defines a metric for the "distance" between two splits of a given dataset.
Later, this will be used to map the landscape and see if there is a smooth, continuous
relationship between datasets' splits' distances and their maximum performance.
"""

import itertools as itt
import multiprocessing as mp
from contextlib import contextmanager, nullcontext
from typing import List, Sequence, Union

import numpy as np

from .triples_factory import TriplesFactory, _tf_cleanup_all
from .utils import calculate_ratios, concatenate_triples_factories, summarize
from ..datasets.base import DataSet, EagerDataset
from ..typing import RandomHint

__all__ = [
    'cleanup_dataset',
    'remix_dataset',
    'remix',
    'deteriorate_dataset',
    'deteriorate',
    'dataset_splits_distance',
    'splits_distance',
    'starmap_ctx',
]


def cleanup_dataset(dataset: DataSet, random_state: RandomHint, randomize_cleanup: bool = False) -> DataSet:
    """Clean up a dataset."""
    triples_groups = _tf_cleanup_all(
        [dataset.training.triples, dataset.testing.triples, dataset.validation.triples],
        random_state=random_state,
        randomize=randomize_cleanup,
    )
    return EagerDataset(*[
        TriplesFactory(
            triples=triples,
            entity_to_id=dataset.training.entity_to_id,
            relation_to_id=dataset.training.relation_to_id,
            compact_id=False,
        )
        for triples in triples_groups
    ])


def remix_dataset(dataset: DataSet, **kwargs) -> DataSet:
    """Remix a dataset."""
    return EagerDataset(*remix(dataset.training, dataset.testing, dataset.validation, **kwargs))


def remix(*triples_factories: TriplesFactory, **kwargs) -> List[TriplesFactory]:
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


def deteriorate_dataset(dataset: DataSet, *, n: Union[int, float], **kwargs) -> DataSet:
    """Remove n triples from the training set and distribute them equally among the testing and validation sets.

    :param dataset: The dataset to deteriorate
    :param n: The number of triples to remove from the training set or ratio if a float is given
    :return: A "deteriorated" dataset

    .. seealso:: :func:`deteriorate`
    """
    return EagerDataset(*deteriorate(dataset.training, dataset.testing, dataset.validation, n=n, **kwargs))


def deteriorate(
    reference: TriplesFactory,
    *others: TriplesFactory,
    n: Union[int, float],
    random_state: RandomHint,
) -> List[TriplesFactory]:
    """Remove n triples from the reference set.

    TODO: take care that triples aren't removed that are the only ones with any given entity
    """
    if isinstance(n, float):
        if n < 0 or 1 <= n:
            raise ValueError
        n = int(n * reference.num_triples)

    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    idx = random_state.choice(reference.num_triples, size=n, replace=False)

    first_idx = [i for i in range(reference.num_triples) if i not in idx]
    first = TriplesFactory(
        triples=reference.triples[first_idx],
        entity_to_id=reference.entity_to_id,
        relation_to_id=reference.relation_to_id,
    )

    split_indices = np.array_split(idx, len(others))
    rest = [
        TriplesFactory(
            triples=np.concatenate([other.triples, reference.triples[split_idx]]),
            entity_to_id=first.entity_to_id,
            relation_to_id=first.relation_to_id,
        )
        for split_idx, other in zip(split_indices, others)
    ]
    return [first, *rest]


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
    return 1 - len(train_1.intersection(train_2)) / len(train_1.union(non_train_1))


def _smt(x):
    return set(tuple(xx) for xx in x)


@contextmanager
def starmap_ctx(use_multiprocessing: bool = False):
    """Create a context that can run `starmap`."""
    try:
        if use_multiprocessing:
            yield mp.Pool(mp.cpu_count() - 1)
        else:
            yield nullcontext(itt)
    finally:
        pass


def _main():
    from pykeen.datasets import Nations
    import numpy as np
    import itertools as itt
    n = Nations()
    summarize(n.training, n.testing, n.validation)

    trials = 35
    splits = [
        remix(
            n.training, n.testing, n.validation,
            random_state=i,
        )
        for i in range(trials)
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
