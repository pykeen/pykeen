# -*- coding: utf-8 -*-

"""Implementation of triples splitting functions."""

import logging
import typing
from abc import abstractmethod
from typing import Collection, Optional, Sequence, Set, Tuple, Type, Union

import numpy
import pandas
import torch
from class_resolver.api import ClassResolver, HintOrType

from ..constants import COLUMN_LABELS
from ..typing import LABEL_HEAD, LABEL_RELATION, LABEL_TAIL, MappedTriples, Target, TorchRandomHint
from ..utils import ensure_torch_random_state

logger = logging.getLogger(__name__)

__all__ = [
    "split",
]


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

    :raises ValueError:
        If the given sizes are different from the number of triples in mapped triples
    """
    num_triples = mapped_triples.shape[0]
    if sum(sizes) != num_triples:
        raise ValueError(f"Received {num_triples} triples, but the sizes sum up to {sum(sizes)}")

    # Split indices
    idx = torch.randperm(num_triples, generator=random_state)
    idx_groups = idx.split(split_size=sizes, dim=0)

    # Split triples
    triples_groups = [mapped_triples[idx] for idx in idx_groups]
    logger.info(
        "done splitting triples to groups of sizes %s",
        [triples.shape[0] for triples in triples_groups],
    )

    return triples_groups


def _get_cover_for_column(df: pandas.DataFrame, column: Target, index_column: str = "index") -> Set[int]:
    return set(df.groupby(by=column).agg({index_column: "min"})[index_column].values)


def _get_covered_entities(df: pandas.DataFrame, chosen: Collection[int]) -> Set[int]:
    return set(numpy.unique(df.loc[df["index"].isin(chosen), [LABEL_HEAD, LABEL_TAIL]]))


def _get_cover_deterministic(triples: MappedTriples) -> torch.BoolTensor:
    """
    Get a coverage mask for all entities and relations.

    The implementation uses a greedy coverage algorithm for selecting triples. If there are multiple triples to
    choose, the smaller ID is preferred.

    1. Select one triple for each relation.
    2. Select one triple for each head entity, which is not yet covered.
    3. Select one triple for each tail entity, which is not yet covered.

    The cover is guaranteed to contain at most $num_relations + num_unique_heads + num_unique_tails$ triples.

    :param triples: shape: (n, 3)
        The triples (ID-based).

    :return: shape: (n,)
        A boolean mask indicating whether the triple is part of the cover.
    """
    df = pandas.DataFrame(data=triples.numpy(), columns=COLUMN_LABELS).reset_index()

    # select one triple per relation
    chosen = _get_cover_for_column(df=df, column=LABEL_RELATION)

    # maintain set of covered entities
    covered = _get_covered_entities(df=df, chosen=chosen)

    # Select one triple for each head/tail entity, which is not yet covered.
    for column in (LABEL_HEAD, LABEL_TAIL):
        covered = _get_covered_entities(df=df, chosen=chosen)
        chosen |= _get_cover_for_column(df=df[~df[column].isin(covered)], column=column)

    # create mask
    num_triples = triples.shape[0]
    seed_mask = torch.zeros(num_triples, dtype=torch.bool)
    seed_mask[list(chosen)] = True
    return seed_mask


class TripleCoverageError(RuntimeError):
    """An exception thrown when not all entities/relations are covered by triples."""

    def __init__(self, arr, name: str = "ids"):
        """
        Initialize the error.

        :param arr: shape: (num_indices,)
            the array of covering triple IDs
        :param name:
            the name to use for creating the error message
        """
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

    :raises ValueError:
        if the ratio sum is bigger than 1.0
    """
    # Prepare split index
    if isinstance(ratios, float):
        ratios = [ratios]
    ratios = tuple(ratios)
    ratio_sum = sum(ratios)
    if ratio_sum < 1.0 - epsilon:
        ratios = ratios + (1.0 - ratio_sum,)
    elif ratio_sum > 1.0 + epsilon:
        raise ValueError(f"ratios sum to more than 1.0: {ratios} (sum={ratio_sum})")
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


class Cleaner:
    """A cleanup method for ensuring that all entities are contained in the triples of the first split part."""

    @abstractmethod
    def cleanup_pair(
        self,
        reference: MappedTriples,
        other: MappedTriples,
        random_state: TorchRandomHint,
    ) -> Tuple[MappedTriples, MappedTriples]:
        """
        Clean up one set of triples with respect to a reference set.

        :param reference:
            the reference set of triples, which shall contain triples for all entities
        :param other:
            the other set of triples
        :param random_state:
            the random state to use, if any randomized operations take place

        :return:
            a pair (reference, other), where some triples of other may have been moved into reference
        """
        raise NotImplementedError

    def __call__(
        self,
        triples_groups: Sequence[MappedTriples],
        random_state: TorchRandomHint,
    ) -> Sequence[MappedTriples]:
        """Cleanup a list of triples array with respect to the first array."""
        reference, *others = triples_groups
        result = []
        for other in others:
            reference, other = self.cleanup_pair(reference=reference, other=other, random_state=random_state)
            result.append(other)
        return reference, *result


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
    :param max_ids:
        The maximum identifier in each column. Calculates it automatically if not given.

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
        max_ids = typing.cast(
            Tuple[int, int],
            tuple(max(training[:, col].max().item(), testing[:, col].max().item()) + 1 for col in columns),
        )
    for col, max_id in zip(columns, max_ids):
        # IDs not in training
        not_in_training_mask = torch.ones(max_id, dtype=torch.bool)
        not_in_training_mask[training[:, col].view(-1)] = False

        # triples with exclusive test IDs
        exclusive_triples = not_in_training_mask[testing[:, col].view(-1)].view(-1, len(col)).any(dim=-1)
        to_move_mask = to_move_mask | exclusive_triples
    return to_move_mask


class RandomizedCleaner(Cleaner):
    """Cleanup a triples array by randomly selecting testing triples and recalculate to minimize moves.

    1. Calculate ``move_id_mask`` as in :func:`_prepare_cleanup`
    2. Choose a triple to move, recalculate ``move_id_mask``
    3. Continue until ``move_id_mask`` has no true bits
    """

    # docstr-coverage: inherited
    def cleanup_pair(
        self,
        reference: MappedTriples,
        other: MappedTriples,
        random_state: TorchRandomHint,
    ) -> Tuple[MappedTriples, MappedTriples]:  # noqa: D102
        generator = ensure_torch_random_state(random_state)
        move_id_mask = _prepare_cleanup(reference, other)

        # While there are still triples that should be moved to the training set
        while move_id_mask.any():
            # Pick a random triple to move over to the training triples
            (candidates,) = move_id_mask.nonzero(as_tuple=True)
            idx = torch.randint(candidates.shape[0], size=(1,), generator=generator)
            idx = candidates[idx]

            # add to training
            reference = torch.cat([reference, other[idx].view(1, -1)], dim=0)
            # remove from testing
            other = torch.cat([other[:idx], other[idx + 1 :]], dim=0)
            # Recalculate the move_id_mask
            move_id_mask = _prepare_cleanup(reference, other)

        return reference, other


class DeterministicCleaner(Cleaner):
    """Cleanup a triples array (testing) with respect to another (training)."""

    # docstr-coverage: inherited
    def cleanup_pair(
        self,
        reference: MappedTriples,
        other: MappedTriples,
        random_state: TorchRandomHint,
    ) -> Tuple[MappedTriples, MappedTriples]:  # noqa: D102
        move_id_mask = _prepare_cleanup(reference, other)
        reference = torch.cat([reference, other[move_id_mask]])
        other = other[~move_id_mask]
        return reference, other


cleaner_resolver: ClassResolver[Cleaner] = ClassResolver.from_subclasses(base=Cleaner, default=DeterministicCleaner)


class Splitter:
    """A method for splitting triples."""

    @abstractmethod
    def split_absolute_size(
        self,
        mapped_triples: MappedTriples,
        sizes: Sequence[int],
        random_state: torch.Generator,
    ) -> Sequence[MappedTriples]:
        """Split triples into clean groups.

        This method partitions the triples, i.e., each triple is in exactly one group. Moreover, it ensures that
        the first group contains all entities at least once.

        :param mapped_triples: shape: (n, 3)
            the ID-based triples
        :param sizes:
            the absolute number of triples for each split part.
        :param random_state:
            the random state used for splitting

        :return:
            a sequence of ID-based triples for each split part. the absolute may be different to ensure the constraint.
        """
        raise NotImplementedError

    def split(
        self,
        *,
        mapped_triples: MappedTriples,
        ratios: Union[float, Sequence[float]] = 0.8,
        random_state: TorchRandomHint = None,
    ) -> Sequence[MappedTriples]:
        """Split triples into clean groups.

        :param mapped_triples: shape: (n, 3)
            the ID-based triples
        :param random_state:
            the random state used to shuffle and split the triples
        :param ratios:
            There are three options for this argument.
            First, a float can be given between 0 and 1.0, non-inclusive. The first set of triples will get this
            ratio and the second will get the rest.
            Second, a list of ratios can be given for which set in which order should get what ratios as
            in ``[0.8, 0.1]``.
            The final ratio can be omitted because that can be calculated.
            Third, all ratios can be explicitly set in order such as in ``[0.8, 0.1, 0.1]`` where the sum of
            all ratios is 1.0.

        :return:
            A partition of triples, which are split (approximately) according to the ratios.
        """
        random_state = ensure_torch_random_state(random_state)
        ratios = normalize_ratios(ratios=ratios)
        sizes = get_absolute_split_sizes(n_total=mapped_triples.shape[0], ratios=ratios)
        triples_groups = self.split_absolute_size(
            mapped_triples=mapped_triples,
            sizes=sizes,
            random_state=random_state,
        )
        for i, (triples, exp_size, exp_ratio) in enumerate(zip(triples_groups, sizes, ratios)):
            actual_size = triples.shape[0]
            actual_ratio = actual_size / exp_size * exp_ratio
            if actual_size != exp_size:
                logger.warning(
                    f"Requested ratio[{i}]={exp_ratio:.3f} (equal to size {exp_size}), but got {actual_ratio:.3f} "
                    f"(equal to size {actual_size}) to ensure that all entities/relations occur in train.",
                )
        return triples_groups


class CleanupSplitter(Splitter):
    """
    The cleanup splitter first randomly splits the triples and then cleans up.

    In the cleanup process, triples are moved into the train part until all entities occur at least once in train.

    The splitter supports two variants of cleanup, cf. ``cleaner_resolver``.
    """

    def __init__(self, cleaner: HintOrType[Cleaner] = None) -> None:
        """
        Initialize the splitter.

        :param cleaner:
            the cleanup method to use. Defaults to the fast deterministic cleaner,
            which may lead to larger deviances between desired and actual triple count.
        """
        self.cleaner = cleaner_resolver.make(cleaner)

    # docstr-coverage: inherited
    def split_absolute_size(
        self,
        mapped_triples: MappedTriples,
        sizes: Sequence[int],
        random_state: torch.Generator,
    ) -> Sequence[MappedTriples]:  # noqa: D102
        triples_groups = _split_triples(
            mapped_triples,
            sizes=sizes,
            random_state=random_state,
        )
        # Make sure that the first element has all the right stuff in it
        logger.debug("cleaning up groups")
        triples_groups = self.cleaner(triples_groups, random_state=random_state)
        logger.debug("done cleaning up groups")
        return triples_groups


class CoverageSplitter(Splitter):
    """This splitter greedily selects training triples such that each entity is covered and then splits the rest."""

    # docstr-coverage: inherited
    def split_absolute_size(
        self,
        mapped_triples: MappedTriples,
        sizes: Sequence[int],
        random_state: torch.Generator,
    ) -> Sequence[MappedTriples]:  # noqa: D102
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
        return [torch.cat([train_seed, train], dim=0), *rest]


splitter_resolver: ClassResolver[Splitter] = ClassResolver.from_subclasses(base=Splitter, default=CoverageSplitter)


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
        The name of the method to use, cf. :data:`splitter_resolver`. Defaults to "coverage".

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
    # backwards compatibility
    splitter_cls: Type[Splitter] = splitter_resolver.lookup(method)
    kwargs = dict()
    if splitter_cls is CleanupSplitter and randomize_cleanup:
        kwargs["cleaner"] = cleaner_resolver.normalize_cls(RandomizedCleaner)
    return splitter_resolver.make(splitter_cls, pos_kwargs=kwargs).split(
        mapped_triples=mapped_triples,
        ratios=ratios,
        random_state=random_state,
    )
