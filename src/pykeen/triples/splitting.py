# -*- coding: utf-8 -*-

"""Implementation of triples splitting functions."""

import logging
import typing
from typing import Optional, Sequence, Tuple, Union

import numpy
import torch

from ..typing import MappedTriples, TorchRandomHint
from ..utils import ensure_torch_random_state

logger = logging.getLogger(__name__)

__all__ = [
    "split",
]

SPLIT_METHODS = (
    'cleanup',
    'coverage',
)


def _split_triples(
    mapped_triples: MappedTriples,
    sizes: Sequence[int],
    random_state: TorchRandomHint = None,
) -> Sequence[MappedTriples]:
    """
    Randomly split triples into groups of given sizes.

    :param mapped_triples: shape: (n, 3)
        The triples.
    :param sizes:
        The sizes.
    :param random_state:
        The random state for reproducible splits.

    :return:
        The splitted triples.
    """
    num_triples = mapped_triples.shape[0]
    if sum(sizes) != num_triples:
        raise ValueError(f"Received {num_triples} triples, but the sizes sum up to {sum(sizes)}")

    # Split indices
    idx = torch.randperm(num_triples, generator=random_state)
    idx_groups = idx.split(split_size=sizes, dim=0)

    # Split triples
    triples_groups = [
        mapped_triples[idx]
        for idx in idx_groups
    ]
    logger.info(
        'done splitting triples to groups of sizes %s',
        [triples.shape[0] for triples in triples_groups],
    )

    return triples_groups


def _get_cover_deterministic(triples: MappedTriples) -> torch.BoolTensor:
    """
    Get a coverage mask for all entities and relations.

    :param triples: shape: (n, 3)
        The triples (ID-based).

    :return: shape: (n,)
        A boolean mask indicating whether the triple is part of the cover.
    """
    num_entities = triples[:, [0, 2]].max() + 1
    num_relations = triples[:, 1].max() + 1
    num_triples = triples.shape[0]

    # index
    entities = torch.full(size=(num_entities,), fill_value=-1, dtype=torch.long)
    relations = torch.full(size=(num_relations,), fill_value=-1, dtype=torch.long)
    h, r, t = triples.T
    triple_id = torch.arange(num_triples)
    entities[h] = relations[r] = entities[t] = triple_id

    if entities.min() < 0:
        raise TripleCoverageError(arr=entities, name="entities")
    if relations.min() < 0:
        raise TripleCoverageError(arr=relations, name="relations")

    # select
    seed_mask = torch.zeros(num_triples, dtype=torch.bool)
    seed_mask[entities] = True
    seed_mask[relations] = True
    return seed_mask


class TripleCoverageError(RuntimeError):
    """An exception thrown when not all entities/relations are covered by triples."""

    def __init__(self, arr, name: str = "ids"):
        r = sorted((arr < 0).nonzero(as_tuple=False))
        super().__init__(
            f"Could not cover the following {name} from the provided triples: {r}. One possible reason is that you are"
            f" working with triples from a non-compact ID mapping, i.e. where the IDs do not span the full range of "
            f"[0, ..., num_ids - 1]",
        )


def normalize_ratios(
    ratios: Union[float, Sequence[float]],
    epsilon: float = 1.0e-06,
) -> Tuple[float, ...]:
    """Normalize relative sizes.

    If the sum is smaller than 1, adds (1 - sum)

    :param ratios:
        The ratios.
    :param epsilon:
        A small constant for comparing sum of ratios against 1.

    :return:
        A sequence of ratios of at least two elements which sums to one.
    """
    # Prepare split index
    if isinstance(ratios, float):
        ratios = [ratios]
    ratios = tuple(ratios)
    ratio_sum = sum(ratios)
    if ratio_sum < 1.0 - epsilon:
        ratios = ratios + (1.0 - ratio_sum,)
    elif ratio_sum > 1.0 + epsilon:
        raise ValueError(f'ratios sum to more than 1.0: {ratios} (sum={ratio_sum})')
    return ratios


def get_absolute_split_sizes(
    n_total: int,
    ratios: Sequence[float],
) -> Tuple[int, ...]:
    """
    Compute absolute sizes of splits from given relative sizes.

    .. note ::
        This method compensates for rounding errors, and ensures that the absolute sizes sum up to the total number.

    :param n_total:
        The total number.
    :param ratios:
        The relative ratios (should sum to 1).

    :return:
        The absolute sizes.
    """
    # due to rounding errors we might lose a few points, thus we use cumulative ratio
    cum_ratio = numpy.cumsum(ratios)
    cum_ratio[-1] = 1.0
    cum_ratio = numpy.r_[numpy.zeros(1), cum_ratio]
    split_points = (cum_ratio * n_total).astype(numpy.int64)
    sizes = numpy.diff(split_points)
    return tuple(sizes)


def _tf_cleanup_all(
    triples_groups: Sequence[MappedTriples],
    *,
    random_state: TorchRandomHint = None,
) -> Sequence[MappedTriples]:
    """Cleanup a list of triples array with respect to the first array."""
    reference, *others = triples_groups
    rv = []
    for other in others:
        if random_state is not None:
            reference, other = _tf_cleanup_randomized(reference, other, random_state)
        else:
            reference, other = _tf_cleanup_deterministic(reference, other)
        rv.append(other)
    # [...] is necessary for Python 3.7 compatibility
    return [reference, *rv]


def _tf_cleanup_deterministic(training: MappedTriples, testing: MappedTriples) -> Tuple[MappedTriples, MappedTriples]:
    """Cleanup a triples array (testing) with respect to another (training)."""
    move_id_mask = _prepare_cleanup(training, testing)
    training = torch.cat([training, testing[move_id_mask]])
    testing = testing[~move_id_mask]
    return training, testing


def _tf_cleanup_randomized(
    training: MappedTriples,
    testing: MappedTriples,
    random_state: TorchRandomHint = None,
) -> Tuple[MappedTriples, MappedTriples]:
    """Cleanup a triples array, but randomly select testing triples and recalculate to minimize moves.

    1. Calculate ``move_id_mask`` as in :func:`_tf_cleanup_deterministic`
    2. Choose a triple to move, recalculate move_id_mask
    3. Continue until move_id_mask has no true bits
    """
    generator = ensure_torch_random_state(random_state)
    move_id_mask = _prepare_cleanup(training, testing)

    # While there are still triples that should be moved to the training set
    while move_id_mask.any():
        # Pick a random triple to move over to the training triples
        candidates, = move_id_mask.nonzero(as_tuple=True)
        idx = torch.randint(candidates.shape[0], size=(1,), generator=generator)
        idx = candidates[idx]

        # add to training
        training = torch.cat([training, testing[idx].view(1, -1)], dim=0)
        # remove from testing
        testing = torch.cat([testing[:idx], testing[idx + 1:]], dim=0)
        # Recalculate the move_id_mask
        move_id_mask = _prepare_cleanup(training, testing)

    return training, testing


def _prepare_cleanup(
    training: MappedTriples,
    testing: MappedTriples,
    max_ids: Optional[Tuple[int, int]] = None,
) -> torch.BoolTensor:
    """
    Calculate a mask for the test triples with triples containing test-only entities or relations.

    :param training: shape: (n, 3)
        The training triples.
    :param testing: shape: (m, 3)
        The testing triples.

    :return: shape: (m,)
        The move mask.
    """
    # base cases
    if len(testing) == 0:
        return torch.empty(0, dtype=torch.bool)
    if len(training) == 0:
        return torch.ones(testing.shape[0], dtype=torch.bool)

    columns = [[0, 2], [1]]
    to_move_mask = torch.zeros(1, dtype=torch.bool)
    if max_ids is None:
        max_ids = typing.cast(Tuple[int, int], tuple(
            max(training[:, col].max().item(), testing[:, col].max().item()) + 1
            for col in columns
        ))
    for col, max_id in zip(columns, max_ids):
        # IDs not in training
        not_in_training_mask = torch.ones(max_id, dtype=torch.bool)
        not_in_training_mask[training[:, col].view(-1)] = False

        # triples with exclusive test IDs
        exclusive_triples = not_in_training_mask[testing[:, col].view(-1)].view(-1, len(col)).any(dim=-1)
        to_move_mask = to_move_mask | exclusive_triples
    return to_move_mask


def split(
    mapped_triples: MappedTriples,
    ratios: Union[float, Sequence[float]] = 0.8,
    random_state: TorchRandomHint = None,
    randomize_cleanup: bool = False,
    method: Optional[str] = None,
) -> Sequence[MappedTriples]:
    """Split triples into clean groups.

    :param mapped_triples: shape: (n, 3)
        The ID-based triples.
    :param ratios:
        There are three options for this argument.
        First, a float can be given between 0 and 1.0, non-inclusive. The first set of triples will get this ratio and
        the second will get the rest.
        Second, a list of ratios can be given for which set in which order should get what ratios as in ``[0.8, 0.1]``.
        The final ratio can be omitted because that can be calculated.
        Third, all ratios can be explicitly set in order such as in ``[0.8, 0.1, 0.1]`` where the sum of all ratios is
        1.0.
    :param random_state:
        The random state used to shuffle and split the triples.
    :param randomize_cleanup:
        If true, uses the non-deterministic method for moving triples to the training set. This has the advantage that
        it does not necessarily have to move all of them, but it might be significantly slower since it moves one
        triple at a time.
    :param method:
        The name of the method to use, from SPLIT_METHODS. Defaults to "coverage".

    :return:
        A partition of triples, which are split (approximately) according to the ratios.

    .. code-block:: python

        ratio = 0.8  # makes a [0.8, 0.2] split
        train, test = split(triples, ratio)

        ratios = [0.8, 0.1]  # makes a [0.8, 0.1, 0.1] split
        train, test, val = split(triples, ratios)

        ratios = [0.8, 0.1, 0.1]  # also makes a [0.8, 0.1, 0.1] split
        train, test, val = split(triples, ratios)
    """
    if method is None:
        method = "coverage"
    if method not in SPLIT_METHODS:
        raise ValueError(f"Invalid split method: \"{method}\". Allowed are {SPLIT_METHODS}")

    random_state = ensure_torch_random_state(random_state)
    ratios = normalize_ratios(ratios=ratios)
    sizes = get_absolute_split_sizes(n_total=mapped_triples.shape[0], ratios=ratios)

    if method == 'cleanup':
        triples_groups = _split_triples(
            mapped_triples,
            sizes=sizes,
            random_state=random_state,
        )
        # Make sure that the first element has all the right stuff in it
        logger.debug('cleaning up groups')
        triples_groups = _tf_cleanup_all(triples_groups, random_state=random_state if randomize_cleanup else None)
        logger.debug('done cleaning up groups')
    elif method == 'coverage' or method is None:
        seed_mask = _get_cover_deterministic(triples=mapped_triples)
        train_seed = mapped_triples[seed_mask]
        remaining_triples = mapped_triples[~seed_mask]
        if train_seed.shape[0] > sizes[0]:
            raise ValueError(f"Could not find a coverage of all entities and relation with only {sizes[0]} triples.")
        remaining_sizes = (sizes[0] - train_seed.shape[0],) + tuple(sizes[1:])
        train, *rest = _split_triples(
            mapped_triples=remaining_triples,
            sizes=remaining_sizes,
            random_state=random_state,
        )
        triples_groups = [torch.cat([train_seed, train], dim=0), *rest]
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
