# -*- coding: utf-8 -*-

"""Implementation of triples splitting functions."""

import logging
import random
from collections import defaultdict
from typing import List, Mapping, Sequence, Set, Tuple

import numpy
import numpy as np

from ..typing import RandomHint
from ..utils import ensure_random_state

logger = logging.getLogger(__name__)


def _cleanup_ratios(ratios):
    # Prepare split index
    if isinstance(ratios, float):
        ratios = [ratios]

    ratio_sum = sum(ratios)
    if ratio_sum == 1.0:
        ratios = ratios[:-1]  # vsplit doesn't take the final number into account.
    elif ratio_sum > 1.0:
        raise ValueError(f'ratios sum to more than 1.0: {ratios} (sum={ratio_sum})')

    return ratios


def _get_group_sizes(triples, ratios):
    # Expects clean ratios!
    n_triples = triples.shape[0]
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
        The sizes. Need to sum up to the number of triples.
    :param random_state:
        The random state for reproducible splits.

    :return:
        The splitted triples.
    """
    # TODO: check size usage
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
    triples,
    ratios,
    random_state: RandomHint = None,
    randomize_cleanup: bool = False,
    method: str = 'old',
):
    """Split the triples into clean groups."""
    random_state = ensure_random_state(random_state)
    ratios = _cleanup_ratios(ratios)
    sizes = _get_group_sizes(triples, ratios)

    if method == 'old':
        triples_groups = _split_triples(
            triples,
            sizes=sizes,
            random_state=random_state,
        )
        # Make sure that the first element has all the right stuff in it
        logger.debug('cleaning up groups')
        triples_groups = _tf_cleanup_all(triples_groups, random_state=random_state if randomize_cleanup else None)
        logger.debug('done cleaning up groups')
    elif method == 'new':
        triples_groups = _split_triples_with_train_coverage(
            triples=triples,
            sizes=sizes,
        )
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


def _split_triples_with_train_coverage(triples: np.ndarray, sizes: Sequence[int]) -> Sequence[np.ndarray]:
    """
    Split triples into groups ensuring that all entities and relations occur in the first group of triples.

    :param triples: shape: (num_triples, 3)
        The triples.
    :param sizes:
        The group sizes.

    :return:
        The groups, where the first group is guaranteed to contain each entity and relation at least once.
    """
    seed_mask = _get_cover_randomized_greedy(triples)
    train_seed = triples[seed_mask]
    remaining_triples = triples[~seed_mask]
    # TODO: what to do if train_seed.shape[0] > sizes[0]
    remaining_sizes = (sizes[0] - train_seed.shape[0],) + tuple(sizes[1:])
    train, *rest = _split_triples(remaining_triples, remaining_sizes)
    return (np.concatenate([train_seed, train]), *rest)


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


def _select_to_cover(
    index: Mapping[int, Sequence[int]],
    covered: Set[int],
    covered_relations: Set[int],
    covered_entities: Set[int],
    triples: np.ndarray,
    seed_mask: np.ndarray,
) -> None:
    for i, triple_ids in index.items():
        if i in covered:
            continue
        # randomly select triple
        tr_id = random.choice(triple_ids)
        seed_mask[tr_id] = True
        # update coverage
        h_id, r_id, t_id = triples[tr_id]
        covered_entities.update(h_id, t_id)
        covered_relations.add(r_id)


def _get_cover_randomized_greedy(triples):
    # TODO: Relative split sizes?
    num_triples = triples.shape[0]
    # index triples
    entities = defaultdict(set)
    relations = defaultdict(set)
    for i, (h, r, t) in enumerate(triples.tolist()):
        entities[h].add(i)
        relations[r].add(i)
        entities[t].add(i)
    # convert to lists; needed for random.choice
    entities = {
        e_id: list(triple_ids)
        for e_id, triple_ids in entities.items()
    }
    relations = {
        r_id: list(triple_ids)
        for r_id, triple_ids in relations.items()
    }
    # randomized greedy cover
    covered_entities = set()
    covered_relations = set()
    seed_mask = numpy.zeros(shape=(num_triples,), dtype=numpy.bool)
    _select_to_cover(
        index=entities,
        covered=covered_entities,
        covered_relations=covered_relations,
        covered_entities=covered_entities,
        triples=triples,
        seed_mask=seed_mask,
    )
    _select_to_cover(
        index=relations,
        covered=covered_relations,
        covered_relations=covered_relations,
        covered_entities=covered_entities,
        triples=triples,
        seed_mask=seed_mask,
    )
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
