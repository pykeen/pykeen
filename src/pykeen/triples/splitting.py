# -*- coding: utf-8 -*-

"""Implementation of triples splitting functions."""

import logging
from typing import List, Optional, Sequence, Tuple, Union

import numpy
import numpy as np

from ..typing import RandomHint
from ..utils import ensure_random_state

logger = logging.getLogger(__name__)

__all__ = [
    "split",
]


def _get_group_sizes(n_triples: int, ratios: Union[float, Sequence[float]]) -> Sequence[int]:
    # Prepare split index
    if isinstance(ratios, float):
        ratios = [ratios]
    ratio_sum = sum(ratios)
    if ratio_sum == 1.0:
        ratios = ratios[:-1]  # vsplit doesn't take the final number into account.
    elif ratio_sum > 1.0:
        raise ValueError(f'ratios sum to more than 1.0: {ratios} (sum={ratio_sum})')

    return [
        int(fraction * n_triples)
        for fraction in ratios
    ]


def _split_triples(
    triples: np.ndarray,
    sizes: Sequence[int],
    random_state: RandomHint = None,
) -> Sequence[np.ndarray]:
    """
    Randomly split triples into groups of given sizes.

    :param triples: shape: (n, 3)
        The triples.
    :param sizes:
        The sizes. TODO: Need to sum up to the number of triples. / need to be at most the number of triples.
    :param random_state:
        The random state for reproducible splits.

    :return:
        The splitted triples.
    """
    n_triples = triples.shape[0]

    # Prepare shuffle index
    idx = np.arange(n_triples)

    random_state = ensure_random_state(random_state)
    random_state.shuffle(idx)

    # Take cumulative sum so the get separated properly
    split_idxs = np.cumsum(sizes)

    # Split triples
    triples_groups = np.vsplit(triples[idx], split_idxs)
    logger.info(
        'done splitting triples to groups of sizes %s',
        [triples.shape[0] for triples in triples_groups],
    )

    return triples_groups


def split(
    triples: np.ndarray,
    ratios: Union[float, Sequence[float]],
    random_state: RandomHint = None,
    randomize_cleanup: bool = False,
    method: Optional[str] = None,
) -> Sequence[np.ndarray]:
    """Split the triples into clean groups."""
    random_state = ensure_random_state(random_state)
    sizes = _get_group_sizes(n_triples=triples.shape[0], ratios=ratios)

    if method == 'cleanup':
        triples_groups = _split_triples(
            triples,
            sizes=sizes,
            random_state=random_state,
        )
        # Make sure that the first element has all the right stuff in it
        logger.debug('cleaning up groups')
        triples_groups = _tf_cleanup_all(triples_groups, random_state=random_state if randomize_cleanup else None)
        logger.debug('done cleaning up groups')
    elif method == 'coverage' or method is None:
        seed_mask = _get_cover_deterministic(triples=triples)
        train_seed = triples[seed_mask]
        remaining_triples = triples[~seed_mask]
        if train_seed.shape[0] > sizes[0]:
            raise ValueError(f"Could not find a coverage of all entities and relation with only {sizes[0]} triples.")
        remaining_sizes = (sizes[0] - train_seed.shape[0],) + tuple(sizes[1:])
        train, *rest = _split_triples(
            triples=remaining_triples,
            sizes=remaining_sizes,
            random_state=random_state,
        )
        result = np.concatenate([train_seed, train]), *rest
        triples_groups = result
    else:
        raise ValueError(f'invalid method: {method}')

    for i, (triples, exp_size, exp_ratio) in enumerate(zip(triples_groups, sizes, ratios)):
        actual_size = triples.shape[0]
        actual_ratio = actual_size / exp_size * exp_ratio
        if actual_size != exp_size:
            logger.warning(
                f'Requested ratio[{i}]={exp_ratio:.3f} (equal to size {exp_size}), but got {actual_ratio:.3f} '
                f'(equal to size {actual_size}) to ensure that all entities/relations occur in train.',
            )

    return triples_groups


def _get_cover_deterministic(triples: np.ndarray) -> np.ndarray:
    """
    Get a coverage mask for all entities and relations.

    :param triples: shape: (n, 3)
        The triples.

    :return: shape: (n,)
        A boolean mask indicating whether the triple is part of the cover.
    """
    num_entities = triples[:, [0, 2]].max() + 1
    num_relations = triples[:, 1].max() + 1
    num_triples = triples.shape[0]

    # index
    entities = numpy.full(shape=(num_entities,), fill_value=-1, dtype=numpy.int64)
    relations = numpy.full(shape=(num_relations,), fill_value=-1, dtype=numpy.int64)
    h, r, t = triples.T
    triple_id = numpy.arange(num_triples)
    entities[h] = relations[r] = entities[t] = triple_id

    if entities.min() < 0:
        raise RuntimeError(f'unfilled entities exist: {entities}')
    if relations.min() < 0:
        raise RuntimeError(f'unfilled relations exist: {relations}')

    # select
    seed_mask = numpy.zeros(shape=(num_triples,), dtype=numpy.bool)
    seed_mask[np.r_[entities, relations]] = True
    return seed_mask


def _tf_cleanup_all(
    triples_groups: Sequence[np.ndarray],
    *,
    random_state: RandomHint = None,
) -> List[np.ndarray]:
    """Cleanup a list of triples array with respect to the first array."""
    reference, *others = triples_groups
    rv = []
    for other in others:
        if random_state is not None:
            reference, other = _tf_cleanup_randomized(reference, other, random_state)
        else:
            reference, other = _tf_cleanup_deterministic(reference, other)
        rv.append(other)
    return [reference, *rv]


def _tf_cleanup_deterministic(training: np.ndarray, testing: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cleanup a triples array (testing) with respect to another (training)."""
    move_id_mask = _prepare_cleanup(training, testing)

    training = np.concatenate([training, testing[move_id_mask]])
    testing = testing[~move_id_mask]

    return training, testing


def _tf_cleanup_randomized(
    training: np.ndarray,
    testing: np.ndarray,
    random_state: RandomHint = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cleanup a triples array, but randomly select testing triples and recalculate to minimize moves.

    1. Calculate ``move_id_mask`` as in :func:`_tf_cleanup_deterministic`
    2. Choose a triple to move, recalculate move_id_mask
    3. Continue until move_id_mask has no true bits
    """
    random_state = ensure_random_state(random_state)

    move_id_mask = _prepare_cleanup(training, testing)

    # While there are still triples that should be moved to the training set
    while move_id_mask.any():
        # Pick a random triple to move over to the training triples
        idx = random_state.choice(move_id_mask.nonzero()[0])
        training = np.concatenate([training, testing[idx].reshape(1, -1)])

        # Recalculate the testing triples without that index
        testing_mask = np.ones_like(move_id_mask)
        testing_mask[idx] = False
        testing = testing[testing_mask]

        # Recalculate the training entities, testing entities, to_move, and move_id_mask
        move_id_mask = _prepare_cleanup(training, testing)

    return training, testing


def _prepare_cleanup(training: np.ndarray, testing: np.ndarray) -> np.ndarray:
    to_move_mask = None
    for col in [[0, 2], 1]:
        training_ids, test_ids = [np.unique(triples[:, col]) for triples in [training, testing]]
        to_move = test_ids[~np.isin(test_ids, training_ids)]
        this_to_move_mask = np.isin(testing[:, col], to_move)
        if this_to_move_mask.ndim > 1:
            this_to_move_mask = this_to_move_mask.any(axis=1)
        if to_move_mask is None:
            to_move_mask = this_to_move_mask
        else:
            to_move_mask = this_to_move_mask | to_move_mask

    return to_move_mask
