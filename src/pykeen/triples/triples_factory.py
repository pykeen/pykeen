# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

import dataclasses
import logging
import os
import re
from collections import Counter
from typing import Collection, Iterable, List, Mapping, Optional, Sequence, Set, TextIO, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .instances import Instances, LCWAInstances, SLCWAInstances
from .utils import load_triples
from ..typing import EntityMapping, LabeledTriples, MappedTriples, RandomHint, RelationMapping
from ..utils import compact_mapping, ensure_random_state, invert_mapping, slice_triples

__all__ = [
    'TriplesFactory',
    'create_entity_mapping',
    'create_relation_mapping',
    'INVERSE_SUFFIX',
]

logger = logging.getLogger(__name__)

INVERSE_SUFFIX = '_inverse'
TRIPLES_DF_COLUMNS = ('head_id', 'head_label', 'relation_id', 'relation_label', 'tail_id', 'tail_label')


def get_unique_entity_ids_from_triples_tensor(mapped_triples: MappedTriples) -> torch.LongTensor:
    """Return the unique entity IDs used in a tensor of triples."""
    return mapped_triples[:, [0, 2]].unique()


def create_entity_mapping(triples: LabeledTriples) -> EntityMapping:
    """Create mapping from entity labels to IDs.

    :param triples: shape: (n, 3), dtype: str
    """
    # Split triples
    heads, tails = triples[:, 0], triples[:, 2]
    # Sorting ensures consistent results when the triples are permuted
    entity_labels = sorted(set(heads).union(tails))
    # Create mapping
    return {
        str(label): i
        for (i, label) in enumerate(entity_labels)
    }


def create_relation_mapping(relations: set) -> RelationMapping:
    """Create mapping from relation labels to IDs.

    :param relations: set
    """
    # Sorting ensures consistent results when the triples are permuted
    relation_labels = sorted(
        set(relations),
        key=lambda x: (re.sub(f'{INVERSE_SUFFIX}$', '', x), x.endswith(f'{INVERSE_SUFFIX}')),
    )
    # Create mapping
    return {
        str(label): i
        for (i, label) in enumerate(relation_labels)
    }


def _map_triples_elements_to_ids(
    triples: LabeledTriples,
    entity_to_id: EntityMapping,
    relation_to_id: RelationMapping,
) -> MappedTriples:
    """Map entities and relations to pre-defined ids."""
    if triples.size == 0:
        logger.warning('Provided empty triples to map.')
        return torch.empty(0, 3, dtype=torch.long)

    heads, relations, tails = slice_triples(triples)

    # When triples that don't exist are trying to be mapped, they get the id "-1"
    entity_getter = np.vectorize(entity_to_id.get)
    head_column = entity_getter(heads, [-1])
    tail_column = entity_getter(tails, [-1])
    relation_getter = np.vectorize(relation_to_id.get)
    relation_column = relation_getter(relations, [-1])

    # Filter all non-existent triples
    head_filter = head_column < 0
    relation_filter = relation_column < 0
    tail_filter = tail_column < 0
    num_no_head = head_filter.sum()
    num_no_relation = relation_filter.sum()
    num_no_tail = tail_filter.sum()

    if (num_no_head > 0) or (num_no_relation > 0) or (num_no_tail > 0):
        logger.warning(
            f"You're trying to map triples with {num_no_head + num_no_tail} entities and {num_no_relation} relations"
            f" that are not in the training set. These triples will be excluded from the mapping.",
        )
        non_mappable_triples = (head_filter | relation_filter | tail_filter)
        head_column = head_column[~non_mappable_triples, None]
        relation_column = relation_column[~non_mappable_triples, None]
        tail_column = tail_column[~non_mappable_triples, None]
        logger.warning(
            f"In total {non_mappable_triples.sum():.0f} from {triples.shape[0]:.0f} triples were filtered out",
        )

    triples_of_ids = np.concatenate([head_column, relation_column, tail_column], axis=1)

    triples_of_ids = np.array(triples_of_ids, dtype=np.long)
    # Note: Unique changes the order of the triples
    # Note: Using unique means implicit balancing of training samples
    unique_mapped_triples = np.unique(ar=triples_of_ids, axis=0)
    return torch.tensor(unique_mapped_triples, dtype=torch.long)


@dataclasses.dataclass
class TriplesFactory:
    """Create instances given the path to triples."""

    #: The mapping from entities' labels to their indices
    entity_to_id: EntityMapping

    #: The mapping from relations' labels to their indices
    relation_to_id: RelationMapping

    # TODO: Deprecation warning. Will be replaced by re-constructing them from ID-based + mapping soon.
    #: A three-column matrix where each row are the head label,
    #: relation label, then tail label
    _triples: LabeledTriples

    #: A three-column matrix where each row are the head identifier,
    #: relation identifier, then tail identifier
    mapped_triples: MappedTriples

    #: A dictionary mapping each relation to its inverse, if inverse triples were created
    relation_to_inverse: Optional[Mapping[str, str]]

    @classmethod
    def from_labeled_triples(
        cls,
        triples: LabeledTriples,
        create_inverse_triples: bool = False,
        entity_to_id: Optional[EntityMapping] = None,
        relation_to_id: Optional[RelationMapping] = None,
        compact_id: bool = True,
    ) -> 'TriplesFactory':
        """
        Create a new triples factory from label-based triples.

        :param triples: shape: (n, 3), dtype: str
            The label-based triples.
        :param create_inverse_triples:
            Whether to create inverse triples.
        :param entity_to_id:
            The mapping from entity labels to ID. If None, create a new one from the triples.
        :param relation_to_id:
            The mapping from relations labels to ID. If None, create a new one from the triples.
        :param compact_id:
            Whether to compact IDs such that the IDs are consecutive.

        :return:
            A new triples factory.
        """
        relations = triples[:, 1]
        unique_relations = set(relations)

        # Check if the triples are inverted already
        relations_already_inverted = cls._check_already_inverted_relations(unique_relations)

        # TODO: invert triples id-based
        if create_inverse_triples or relations_already_inverted:
            create_inverse_triples = True
            if relations_already_inverted:
                logger.info(
                    f'Some triples already have suffix {INVERSE_SUFFIX}. '
                    f'Creating TriplesFactory based on inverse triples',
                )
                relation_to_inverse = {
                    re.sub('_inverse$', '', relation): f"{re.sub('_inverse$', '', relation)}{INVERSE_SUFFIX}"
                    for relation in unique_relations
                }

            else:
                relation_to_inverse = {
                    relation: f"{relation}{INVERSE_SUFFIX}"
                    for relation in unique_relations
                }
                inverse_triples = np.stack(
                    [
                        triples[:, 2],
                        np.array([relation_to_inverse[relation] for relation in relations], dtype=np.str),
                        triples[:, 0],
                    ],
                    axis=-1,
                )
                # extend original triples with inverse ones
                triples = np.concatenate([triples, inverse_triples], axis=0)

        else:
            create_inverse_triples = False
            relation_to_inverse = None

        # Generate entity mapping if necessary
        if entity_to_id is None:
            entity_to_id = create_entity_mapping(triples=triples)
        if compact_id:
            entity_to_id = compact_mapping(mapping=entity_to_id)[0]

        # Generate relation mapping if necessary
        if relation_to_id is None:
            if create_inverse_triples:
                relation_to_id = create_relation_mapping(
                    set(relation_to_inverse.keys()).union(set(relation_to_inverse.values())),
                )
            else:
                relation_to_id = create_relation_mapping(unique_relations)
        if compact_id:
            relation_to_id = compact_mapping(mapping=relation_to_id)[0]

        # Map triples of labels to triples of IDs.
        mapped_triples = _map_triples_elements_to_ids(
            triples=triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

        return cls(
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            _triples=triples,
            mapped_triples=mapped_triples,
            relation_to_inverse=relation_to_inverse,
        )

    @classmethod
    def from_path(
        cls,
        path: Union[str, TextIO],
        create_inverse_triples: bool = False,
        entity_to_id: Optional[EntityMapping] = None,
        relation_to_id: Optional[RelationMapping] = None,
        compact_id: bool = True,
    ) -> 'TriplesFactory':
        """
        Create a new triples factory from triples stored in a file.

        :param path:
            The path where the label-based triples are stored.
        :param create_inverse_triples:
            Whether to create inverse triples.
        :param entity_to_id:
            The mapping from entity labels to ID. If None, create a new one from the triples.
        :param relation_to_id:
            The mapping from relations labels to ID. If None, create a new one from the triples.
        :param compact_id:
            Whether to compact IDs such that the IDs are consecutive.

        :return:
            A new triples factory.
        """
        if isinstance(path, str):
            path = os.path.abspath(path)
        elif isinstance(path, TextIO):
            path = os.path.abspath(path.name)
        else:
            raise TypeError(f'path is invalid type: {type(path)}')

        # TODO: Check if lazy evaluation would make sense
        triples = load_triples(path)

        return cls.from_labeled_triples(
            triples=triples,
            create_inverse_triples=create_inverse_triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            compact_id=compact_id,
        )

    @property
    def create_inverse_triples(self) -> bool:  # noqa: D401
        """Whether inverse triples are added."""
        return self.relation_to_inverse is not None

    @property
    def num_entities(self) -> int:  # noqa: D401
        """The number of unique entities."""
        return len(self.entity_to_id)

    @property
    def num_relations(self) -> int:  # noqa: D401
        """The number of unique relations."""
        return len(self.relation_to_id)

    @property
    def num_triples(self) -> int:  # noqa: D401
        """The number of triples."""
        return self.mapped_triples.shape[0]

    @property
    def triples(self) -> np.ndarray:  # noqa: D401
        """The labeled triples."""
        # TODO: Deprecation warning. Will be replaced by re-constructing them from ID-based + mapping soon.
        return self._triples

    @property
    def entity_id_to_label(self) -> Mapping[int, str]:  # noqa: D401
        """The mapping from entity IDs to their labels."""
        return invert_mapping(mapping=self.entity_to_id)

    @property
    def relation_id_to_label(self) -> Mapping[int, str]:  # noqa: D401
        """The mapping from relation IDs to their labels."""
        return invert_mapping(mapping=self.relation_to_id)

    def get_inverse_relation_id(self, relation: str) -> int:
        """Get the inverse relation identifier for the given relation."""
        if not self.create_inverse_triples:
            raise ValueError('Can not get inverse triple, they have not been created.')
        inverse_relation = self.relation_to_inverse[relation]
        return self.relation_to_id[inverse_relation]

    def extra_repr(self) -> str:
        """Extra representation string."""
        return (
            f"num_entities={self.num_entities}, "
            f"num_relations={self.num_relations}, "
            f"num_triples={self.num_triples}, "
            f"inverse_triples={self.create_inverse_triples}"
        )

    def __repr__(self):  # noqa: D105
        return f'{self.__class__.__name__}({self.extra_repr()})'

    @staticmethod
    def _check_already_inverted_relations(relations: Iterable[str]) -> bool:
        for relation in relations:
            if relation.endswith(INVERSE_SUFFIX):
                # We can terminate the search after finding the first inverse occurrence
                return True

        return False

    def create_slcwa_instances(self) -> Instances:
        """Create sLCWA instances for this factory's triples."""
        return SLCWAInstances(mapped_triples=self.mapped_triples)

    def create_lcwa_instances(self, use_tqdm: Optional[bool] = None) -> Instances:
        """Create LCWA instances for this factory's triples."""
        return LCWAInstances.from_triples(mapped_triples=self.mapped_triples, num_entities=self.num_entities)

    def map_triples_to_id(self, triples: Union[str, LabeledTriples]) -> MappedTriples:
        """Load triples and map to ids based on the existing id mappings of the triples factory.

        Works from either the path to a file containing triples given as string or a numpy array containing triples.
        """
        if isinstance(triples, str):
            triples = load_triples(triples)
        # Ensure 2d array in case only one triple was given
        triples = np.atleast_2d(triples)
        # FIXME this function is only ever used in tests
        return _map_triples_elements_to_ids(
            triples=triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
        )

    def split(
        self,
        ratios: Union[float, Sequence[float]] = 0.8,
        *,
        random_state: RandomHint = None,
        randomize_cleanup: bool = False,
    ) -> List['TriplesFactory']:
        """Split a triples factory into a train/test.

        :param ratios: There are three options for this argument. First, a float can be given between 0 and 1.0,
         non-inclusive. The first triples factory will get this ratio and the second will get the rest. Second,
         a list of ratios can be given for which factory in which order should get what ratios as in ``[0.8, 0.1]``.
         The final ratio can be omitted because that can be calculated. Third, all ratios can be explicitly set in
         order such as in ``[0.8, 0.1, 0.1]`` where the sum of all ratios is 1.0.
        :param random_state: The random state used to shuffle and split the triples in this factory.
        :param randomize_cleanup: If true, uses the non-deterministic method for moving triples to the training set.
         This has the advantage that it doesn't necessarily have to move all of them, but it might be slower.

        .. code-block:: python

            ratio = 0.8  # makes a [0.8, 0.2] split
            training_factory, testing_factory = factory.split(ratio)

            ratios = [0.8, 0.1]  # makes a [0.8, 0.1, 0.1] split
            training_factory, testing_factory, validation_factory = factory.split(ratios)

            ratios = [0.8, 0.1, 0.1]  # also makes a [0.8, 0.1, 0.1] split
            training_factory, testing_factory, validation_factory = factory.split(ratios)
        """
        n_triples = self.triples.shape[0]

        # Prepare shuffle index
        idx = np.arange(n_triples)
        random_state = ensure_random_state(random_state)
        random_state.shuffle(idx)

        # Prepare split index
        if isinstance(ratios, float):
            ratios = [ratios]

        ratio_sum = sum(ratios)
        if ratio_sum == 1.0:
            ratios = ratios[:-1]  # vsplit doesn't take the final number into account.
        elif ratio_sum > 1.0:
            raise ValueError(f'ratios sum to more than 1.0: {ratios} (sum={ratio_sum})')

        sizes = [
            int(split_ratio * n_triples)
            for split_ratio in ratios
        ]
        # Take cumulative sum so the get separated properly
        split_idxs = np.cumsum(sizes)

        # Split triples
        triples_groups = np.vsplit(self.triples[idx], split_idxs)
        logger.info(
            'done splitting triples to groups of sizes %s',
            [triples.shape[0] for triples in triples_groups],
        )

        # Make sure that the first element has all the right stuff in it
        logger.debug('cleaning up groups')
        triples_groups = _tf_cleanup_all(triples_groups, random_state=random_state if randomize_cleanup else None)
        logger.debug('done cleaning up groups')

        for i, (triples, exp_size, exp_ratio) in enumerate(zip(triples_groups, sizes, ratios)):
            actual_size = triples.shape[0]
            actual_ratio = actual_size / exp_size * exp_ratio
            if actual_size != exp_size:
                logger.warning(
                    f'Requested ratio[{i}]={exp_ratio:.3f} (equal to size {exp_size}), but got {actual_ratio:.3f} '
                    f'(equal to size {actual_size}) to ensure that all entities/relations occur in train.',
                )

        # Make new triples factories for each group
        return [
            TriplesFactory.from_labeled_triples(
                triples=triples,
                entity_to_id=self.entity_to_id,
                relation_to_id=self.relation_to_id,
                compact_id=False,
            )
            for triples in triples_groups
        ]

    def get_most_frequent_relations(self, n: Union[int, float]) -> Set[str]:
        """Get the n most frequent relations.

        :param n: Either the (integer) number of top relations to keep or the (float) percentage of top relationships
         to keep
        """
        logger.info(f'applying cutoff of {n} to {self}')
        if isinstance(n, float):
            assert 0 < n < 1
            n = int(self.num_relations * n)
        elif not isinstance(n, int):
            raise TypeError('n must be either an integer or a float')

        counter = Counter(self.triples[:, 1])
        return {
            relation
            for relation, _ in counter.most_common(n)
        }

    def get_idx_for_entities(self, entities: Collection[str], invert: bool = False):
        """Get np.array indices for triples with the given entities."""
        entities = np.asanyarray(entities, dtype=self.triples.dtype)
        return (
            np.isin(self.triples[:, 0], entities, invert=invert)
            & np.isin(self.triples[:, 2], entities, invert=invert)
        )

    def get_idx_for_relations(self, relations: Collection[str], invert: bool = False):
        """Get np.array indices for triples with the given relations."""
        return np.isin(self.triples[:, 1], list(relations), invert=invert)

    def get_triples_for_relations(self, relations: Collection[str], invert: bool = False) -> LabeledTriples:
        """Get the labeled triples containing the given relations."""
        return self.triples[self.get_idx_for_relations(relations, invert=invert)]

    def new_with_relations(self, relations: Collection[str]) -> 'TriplesFactory':
        """Make a new triples factory only keeping the given relations."""
        idx = self.get_idx_for_relations(relations)
        logger.info(
            f'keeping {len(relations)}/{self.num_relations} relations'
            f' and {idx.sum()}/{self.num_triples} triples in {self}',
        )
        return TriplesFactory.from_labeled_triples(triples=self.triples[idx])

    def new_without_relations(self, relations: Collection[str]) -> 'TriplesFactory':
        """Make a new triples factory without the given relations."""
        idx = self.get_idx_for_relations(relations, invert=True)
        logger.info(
            f'removing {len(relations)}/{self.num_relations} relations'
            f' and {idx.sum()}/{self.num_triples} triples',
        )
        return TriplesFactory.from_labeled_triples(triples=self.triples[idx])

    def entity_word_cloud(self, top: Optional[int] = None):
        """Make a word cloud based on the frequency of occurrence of each entity in a Jupyter notebook.

        :param top: The number of top entities to show. Defaults to 100.

        .. warning::

            This function requires the ``word_cloud`` package. Use ``pip install pykeen[plotting]`` to
            install it automatically, or install it yourself with
            ``pip install git+https://github.com/kavgan/word_cloud.git``.
        """
        text = [f'{h} {t}' for h, _, t in self.triples]
        return self._word_cloud(text=text, top=top or 100)

    def relation_word_cloud(self, top: Optional[int] = None):
        """Make a word cloud based on the frequency of occurrence of each relation in a Jupyter notebook.

        :param top: The number of top relations to show. Defaults to 100.

        .. warning::

            This function requires the ``word_cloud`` package. Use ``pip install pykeen[plotting]`` to
            install it automatically, or install it yourself with
            ``pip install git+https://github.com/kavgan/word_cloud.git``.
        """
        text = [r for _, r, _ in self.triples]
        return self._word_cloud(text=text, top=top or 100)

    def _word_cloud(self, *, text: List[str], top: int):
        try:
            from word_cloud.word_cloud_generator import WordCloud
        except ImportError:
            logger.warning(
                'Could not import module `word_cloud`. '
                'Try installing it with `pip install git+https://github.com/kavgan/word_cloud.git`',
            )
            return

        from IPython.core.display import HTML
        word_cloud = WordCloud()
        return HTML(word_cloud.get_embed_code(text=text, topn=top))

    def tensor_to_df(
        self,
        tensor: torch.LongTensor,
        **kwargs: Union[torch.Tensor, np.ndarray, Sequence],
    ) -> pd.DataFrame:
        """Take a tensor of triples and make a pandas dataframe with labels.

        :param tensor: shape: (n, 3)
            The triples, ID-based and in format (head_id, relation_id, tail_id).
        :param kwargs:
            Any additional number of columns. Each column needs to be of shape (n,). Reserved column names:
            {"head_id", "head_label", "relation_id", "relation_label", "tail_id", "tail_label"}.
        :return:
            A dataframe with n rows, and 6 + len(kwargs) columns.
        """
        # Input validation
        additional_columns = set(kwargs.keys())
        forbidden = additional_columns.intersection(TRIPLES_DF_COLUMNS)
        if len(forbidden) > 0:
            raise ValueError(
                f'The key-words for additional arguments must not be in {TRIPLES_DF_COLUMNS}, but {forbidden} were '
                f'used.',
            )

        # convert to numpy
        tensor = tensor.cpu().numpy()
        data = dict(zip(['head_id', 'relation_id', 'tail_id'], tensor.T))

        # vectorized label lookup
        entity_id_to_label = np.vectorize(self.entity_id_to_label.__getitem__)
        relation_id_to_label = np.vectorize(self.relation_id_to_label.__getitem__)
        for column, id_to_label in dict(
            head=entity_id_to_label,
            relation=relation_id_to_label,
            tail=entity_id_to_label,
        ).items():
            data[f'{column}_label'] = id_to_label(data[f'{column}_id'])

        # Additional columns
        for key, values in kwargs.items():
            # convert PyTorch tensors to numpy
            if torch.is_tensor(values):
                values = values.cpu().numpy()
            data[key] = values

        # convert to dataframe
        rv = pd.DataFrame(data=data)

        # Re-order columns
        columns = list(TRIPLES_DF_COLUMNS) + sorted(set(rv.columns).difference(TRIPLES_DF_COLUMNS))
        return rv.loc[:, columns]

    def new_with_restriction(
        self,
        entities: Optional[Collection[str]] = None,
        relations: Optional[Collection[str]] = None,
    ) -> 'TriplesFactory':
        """Make a new triples factory only keeping the given entities and relations, but keeping the ID mapping.

        :param entities:
            The entities of interest. If None, defaults to all entities.
        :param relations:
            The relations of interest. If None, defaults to all relations.

        :return:
            A new triples factory, which has only a subset of the triples containing the entities and relations of
            interest. The label-to-ID mapping is *not* modified.
        """
        if self.create_inverse_triples and relations is not None:
            logger.info(
                'Since %s already contain inverse relations, the relation filter is expanded to contain the inverse '
                'relations as well.',
                str(self),
            )
            relations = list(relations) + list(map(self.relation_to_inverse.__getitem__, relations))

        keep_mask = None

        # Filter for entities
        if entities is not None:
            keep_mask = self.get_idx_for_entities(entities=entities)
            logger.info('Keeping %d/%d entities', len(entities), self.num_entities)

        # Filter for relations
        if relations is not None:
            relation_mask = self.get_idx_for_relations(relations=relations)
            logger.info('Keeping %d/%d relations', len(relations), self.num_relations)
            keep_mask = relation_mask if keep_mask is None else keep_mask & relation_mask

        # No filtering happened
        if keep_mask is None:
            return self

        logger.info('Keeping %d/%d triples', keep_mask.sum(), self.num_triples)
        factory = TriplesFactory.from_labeled_triples(
            triples=self.triples[keep_mask],
            create_inverse_triples=False,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            compact_id=False,
        )

        # manually copy the inverse relation mappings
        if self.create_inverse_triples:
            factory.relation_to_inverse = self.relation_to_inverse

        return factory


def _tf_cleanup_all(
    triples_groups: List[np.ndarray],
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
