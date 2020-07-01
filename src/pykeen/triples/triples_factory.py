# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

import logging
import os
import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Set, TextIO, Tuple, Union

import numpy as np
import torch

from .instances import LCWAInstances, SLCWAInstances
from .utils import load_triples
from ..tqdmw import tqdm
from ..typing import EntityMapping, LabeledTriples, MappedTriples, RelationMapping
from ..utils import compact_mapping, slice_triples

__all__ = [
    'TriplesFactory',
    'create_entity_mapping',
    'create_relation_mapping',
    'INVERSE_SUFFIX',
]

logger = logging.getLogger(__name__)

INVERSE_SUFFIX = '_inverse'


def _create_multi_label_tails_instance(
    mapped_triples: MappedTriples,
    use_tqdm: Optional[bool] = None
) -> Dict[Tuple[int, int], List[int]]:
    """Create for each (h,r) pair the multi tail label."""
    logger.debug('Creating multi label tails instance')

    '''
    The mapped triples matrix has to be a numpy array to ensure correct pair hashing, as explained in
    https://github.com/pykeen/pykeen/commit/1bc71fe4eb2f24190425b0a4d0b9d6c7b9c4653a
    '''
    mapped_triples = mapped_triples.cpu().detach().numpy()

    s_p_to_multi_tails_new = _create_multi_label_instances(
        mapped_triples,
        element_1_index=0,
        element_2_index=1,
        label_index=2,
        use_tqdm=use_tqdm
    )

    logger.debug('Created multi label tails instance')

    return s_p_to_multi_tails_new


def _create_multi_label_instances(
    mapped_triples: MappedTriples,
    element_1_index: int,
    element_2_index: int,
    label_index: int,
    use_tqdm: Optional[bool] = None,
) -> Dict[Tuple[int, int], List[int]]:
    """Create for each (element_1, element_2) pair the multi-label."""
    instance_to_multi_label = defaultdict(set)

    if use_tqdm is None:
        use_tqdm = True

    it = mapped_triples
    if use_tqdm:
        it = tqdm(mapped_triples, unit='triple', unit_scale=True, desc='Grouping triples')
    for row in it:
        instance_to_multi_label[row[element_1_index], row[element_2_index]].add(row[label_index])

    # Create lists out of sets for proper numpy indexing when loading the labels
    # TODO is there a need to have a canonical sort order here?
    instance_to_multi_label_new = {
        key: list(value)
        for key, value in instance_to_multi_label.items()
    }

    return instance_to_multi_label_new


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


class TriplesFactory:
    """Create instances given the path to triples."""

    #: The mapping from entities' labels to their indexes
    entity_to_id: EntityMapping

    #: The mapping from relations' labels to their indexes
    relation_to_id: RelationMapping

    #: A three-column matrix where each row are the head label,
    #: relation label, then tail label
    triples: LabeledTriples

    #: A three-column matrix where each row are the head identifier,
    #: relation identifier, then tail identifier
    mapped_triples: MappedTriples

    #: A dictionary mapping each relation to its inverse, if inverse triples were created
    relation_to_inverse: Optional[Mapping[str, str]]

    def __init__(
        self,
        *,
        path: Union[None, str, TextIO] = None,
        triples: Optional[LabeledTriples] = None,
        create_inverse_triples: bool = False,
        entity_to_id: Optional[EntityMapping] = None,
        relation_to_id: Optional[RelationMapping] = None,
        compact_id: bool = True,
    ) -> None:
        """Initialize the triples factory.

        :param path: The path to a 3-column TSV file with triples in it. If not specified,
         you should specify ``triples``.
        :param triples:  A 3-column numpy array with triples in it. If not specified,
         you should specify ``path``
        :param create_inverse_triples: Should inverse triples be created? Defaults to False.
        :param compact_id:
            Whether to compact the IDs such that they range from 0 to (num_entities or num_relations)-1
        """
        if path is None and triples is None:
            raise ValueError('Must specify either triples or path')
        elif path is not None and triples is not None:
            raise ValueError('Must not specify both triples and path')
        elif path is not None:
            if isinstance(path, str):
                self.path = os.path.abspath(path)
            elif isinstance(path, TextIO):
                self.path = os.path.abspath(path.name)
            else:
                raise TypeError(f'path is invalid type: {type(path)}')

            # TODO: Check if lazy evaluation would make sense
            self.triples = load_triples(path)
        else:  # triples is not None
            self.path = '<None>'
            self.triples = triples

        self._num_entities = len(set(self.triples[:, 0]).union(self.triples[:, 2]))

        relations = self.triples[:, 1]
        unique_relations = set(relations)

        # Check if the triples are inverted already
        relations_already_inverted = self._check_already_inverted_relations(unique_relations)

        if create_inverse_triples or relations_already_inverted:
            self.create_inverse_triples = True
            if relations_already_inverted:
                logger.info(
                    f'Some triples already have suffix {INVERSE_SUFFIX}. '
                    f'Creating TriplesFactory based on inverse triples')
                self.relation_to_inverse = {
                    re.sub('_inverse$', '', relation): f"{re.sub('_inverse$', '', relation)}{INVERSE_SUFFIX}"
                    for relation in unique_relations
                }

            else:
                self.relation_to_inverse = {
                    relation: f"{relation}{INVERSE_SUFFIX}"
                    for relation in unique_relations
                }
                inverse_triples = np.stack(
                    [
                        self.triples[:, 2],
                        np.array([self.relation_to_inverse[relation] for relation in relations], dtype=np.str),
                        self.triples[:, 0],
                    ],
                    axis=-1,
                )
                # extend original triples with inverse ones
                self.triples = np.concatenate([self.triples, inverse_triples], axis=0)
                self._num_relations = 2 * len(unique_relations)

        else:
            self.create_inverse_triples = False
            self.relation_to_inverse = None
            self._num_relations = len(unique_relations)

        # Generate entity mapping if necessary
        if entity_to_id is None:
            entity_to_id = create_entity_mapping(triples=self.triples)
        if compact_id:
            entity_to_id = compact_mapping(mapping=entity_to_id)[0]
        self.entity_to_id = entity_to_id

        # Generate relation mapping if necessary
        if relation_to_id is None:
            if self.create_inverse_triples:
                relation_to_id = create_relation_mapping(
                    set(self.relation_to_inverse.keys()).union(set(self.relation_to_inverse.values())),
                )
            else:
                relation_to_id = create_relation_mapping(unique_relations)
        if compact_id:
            relation_to_id = compact_mapping(mapping=relation_to_id)[0]
        self.relation_to_id = relation_to_id

        # Map triples of labels to triples of IDs.
        self.mapped_triples = _map_triples_elements_to_ids(
            triples=self.triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
        )

    @property
    def num_entities(self) -> int:  # noqa: D401
        """The number of unique entities."""
        return self._num_entities

    @property
    def num_relations(self) -> int:  # noqa: D401
        """The number of unique relations."""
        return self._num_relations

    @property
    def num_triples(self) -> int:  # noqa: D401
        """The number of triples."""
        return self.mapped_triples.shape[0]

    def get_inverse_relation_id(self, relation: str) -> int:
        """Get the inverse relation identifier for the given relation."""
        if not self.create_inverse_triples:
            raise ValueError('Can not get inverse triple, they have not been created.')
        inverse_relation = self.relation_to_inverse[relation]
        return self.relation_to_id[inverse_relation]

    def __repr__(self):  # noqa: D105
        return f'{self.__class__.__name__}(path="{self.path}")'

    @staticmethod
    def _check_already_inverted_relations(relations: Iterable[str]) -> bool:
        for relation in relations:
            if relation.endswith(INVERSE_SUFFIX):
                # We can terminate the search after finding the first inverse occurrence
                return True

        return False

    def create_slcwa_instances(self) -> SLCWAInstances:
        """Create sLCWA instances for this factory's triples."""
        return SLCWAInstances(
            mapped_triples=self.mapped_triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
        )

    def create_lcwa_instances(self, use_tqdm: Optional[bool] = None) -> LCWAInstances:
        """Create LCWA instances for this factory's triples."""
        s_p_to_multi_tails = _create_multi_label_tails_instance(
            mapped_triples=self.mapped_triples,
            use_tqdm=use_tqdm,
        )
        sp, multi_o = zip(*s_p_to_multi_tails.items())
        mapped_triples: torch.LongTensor = torch.tensor(sp, dtype=torch.long)
        labels = np.array([np.array(item) for item in multi_o])

        return LCWAInstances(
            mapped_triples=mapped_triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            labels=labels,
        )

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
        random_state: Union[None, int, np.random.RandomState] = None,
    ) -> List['TriplesFactory']:
        """Split a triples factory into a train/test.

        :param ratios: There are three options for this argument. First, a float can be given between 0 and 1.0,
         non-inclusive. The first triples factory will get this ratio and the second will get the rest. Second,
         a list of ratios can be given for which factory in which order should get what ratios as in ``[0.8, 0.1]``.
         The final ratio can be omitted because that can be calculated. Third, all ratios can be explicitly set in
         order such as in ``[0.8, 0.1, 0.1]`` where the sum of all ratios is 1.0.
        :param random_state: The random state used to shuffle and split the triples in this factory.

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
        if random_state is None:
            random_state = np.random.randint(0, 2 ** 32 - 1)
            logger.warning(f'Using random_state={random_state} to split {self}')
        if isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        random_state.shuffle(idx)

        # Prepare split index
        if isinstance(ratios, float):
            ratios = [ratios]

        ratio_sum = sum(ratios)
        if ratio_sum == 1.0:
            ratios = ratios[:-1]  # vsplit doesn't take the final number into account.
        elif ratio_sum > 1.0:
            raise ValueError(f'ratios sum to more than 1.0: {ratios} (sum={ratio_sum})')

        split_idxs = [
            int(split_ratio * n_triples)
            for split_ratio in ratios
        ]
        # Take cumulative sum so the get separated properly
        split_idxs = np.cumsum(split_idxs)

        # Split triples
        triples_groups = np.vsplit(self.triples[idx], split_idxs)
        logger.info(f'split triples to groups of sizes {[triples.shape[0] for triples in triples_groups]}')

        # Make new triples factories for each group
        return [
            TriplesFactory(
                triples=triples,
                entity_to_id=deepcopy(self.entity_to_id),
                relation_to_id=deepcopy(self.relation_to_id),
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

    def get_idx_for_relations(self, relations: Collection[str], invert: bool = False):
        """Get an np.array index for triples with the given relations."""
        return np.isin(self.triples[:, 1], list(relations), invert=invert)

    def get_triples_for_relations(self, relations: Collection[str], invert: bool = False) -> LabeledTriples:
        """Get the labeled triples containing the given relations."""
        return self.triples[self.get_idx_for_relations(relations, invert=invert)]

    def new_with_relations(self, relations: Collection[str]) -> 'TriplesFactory':
        """Make a new triples factory only keeping the given relations."""
        idx = self.get_idx_for_relations(relations)
        logger.info(f'keeping {len(relations)}/{self.num_relations} relations'
                    f' and {idx.sum()}/{self.num_triples} triples in {self}')
        return TriplesFactory(triples=self.triples[idx])

    def new_without_relations(self, relations: Collection[str]) -> 'TriplesFactory':
        """Make a new triples factory without the given relations."""
        idx = self.get_idx_for_relations(relations, invert=True)
        logger.info(f'removing {len(relations)}/{self.num_relations} relations'
                    f' and {idx.sum()}/{self.num_triples} triples')
        return TriplesFactory(triples=self.triples[idx])

    def entity_word_cloud(self, top: Optional[int] = None):
        """Make a word cloud based on the frequency of occurrence of each entity in a Jupyter notebook.

        :param top: The number of top entities to show. Defaults to 100.

        .. warning::

            The ``word_cloud`` package is not available through pip. You should install it yourself
            with ``pip install git+ssh://git@github.com/kavgan/word_cloud.git``.
        """
        text = [f'{h} {t}' for h, _, t in self.triples]
        return self._word_cloud(text=text, top=top or 100)

    def relation_word_cloud(self, top: Optional[int] = None):
        """Make a word cloud based on the frequency of occurrence of each relation in a Jupyter notebook.

        :param top: The number of top relations to show. Defaults to 100.

        .. warning::

            The ``word_cloud`` package is not available through pip. You should install it yourself
            with ``pip install git+ssh://git@github.com/kavgan/word_cloud.git``.
        """
        text = [r for _, r, _ in self.triples]
        return self._word_cloud(text=text, top=top or 100)

    def _word_cloud(self, *, text: List[str], top: int):
        try:
            from word_cloud.word_cloud_generator import WordCloud
        except ImportError:
            logger.warning(
                'Could not import module `word_cloud`. '
                'Try installing with `pip install git+ssh://git@github.com/kavgan/word_cloud.git`',
            )
            return

        from IPython.core.display import HTML
        word_cloud = WordCloud()
        return HTML(word_cloud.get_embed_code(text=text, topn=top))
