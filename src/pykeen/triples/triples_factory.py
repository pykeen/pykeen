# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

import dataclasses
import itertools
import logging
import os
import re
from typing import Callable, Collection, List, Mapping, Optional, Sequence, Set, TextIO, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .instances import Instances, LCWAInstances, SLCWAInstances
from .utils import load_triples
from ..typing import EntityMapping, LabeledTriples, MappedTriples, RelationMapping, TorchRandomHint
from ..utils import compact_mapping, ensure_torch_random_state, invert_mapping, slice_triples, torch_is_in_1d

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


def _get_triple_mask(
    ids: Collection[int],
    triples: MappedTriples,
    columns: Union[int, Collection[int]],
    invert: bool = False,
    max_id: Optional[int] = None,
) -> torch.BoolTensor:
    # normalize input
    triples = triples[:, columns]
    if isinstance(columns, int):
        columns = [columns]
    mask = torch_is_in_1d(
        query_tensor=triples,
        test_tensor=ids,
        max_id=max_id,
        invert=invert,
    )
    if len(columns) > 1:
        mask = mask.all(dim=-1)
    return mask


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
    cum_ratio = np.cumsum(ratios)
    cum_ratio[-1] = 1.0
    cum_ratio = np.r_[np.zeros(1), cum_ratio]
    split_points = (cum_ratio * n_total).astype(np.int64)
    sizes = np.diff(split_points)
    return tuple(sizes)


def _ensure_ids(
    labels_or_ids: Union[Collection[int], Collection[str]],
    label_to_id: Mapping[str, int],
) -> Collection[int]:
    """Convert labels to IDs."""
    return [
        label_to_id[l_or_i] if isinstance(l_or_i, str) else l_or_i
        for l_or_i in labels_or_ids
    ]


@dataclasses.dataclass
class TriplesFactory:
    """Create instances given the path to triples."""

    #: The mapping from entities' labels to their indices
    entity_to_id: EntityMapping

    #: The mapping from relations' labels to their indices
    relation_to_id: RelationMapping

    #: A three-column matrix where each row are the head identifier,
    #: relation identifier, then tail identifier
    mapped_triples: MappedTriples

    #: Whether to create inverse triples
    create_inverse_triples: bool = False

    # The following fields get generated automatically

    #: The inverse mapping for entity_label_to_id; initialized automatically
    entity_id_to_label: Mapping[int, str] = dataclasses.field(init=False)

    #: The inverse mapping for relation_label_to_id; initialized automatically
    relation_id_to_label: Mapping[int, str] = dataclasses.field(init=False)

    #: A vectorized version of entity_label_to_id; initialized automatically
    _vectorized_entity_mapper: Callable[..., np.ndarray] = dataclasses.field(init=False)

    #: A vectorized version of relation_label_to_id; initialized automatically
    _vectorized_relation_mapper: Callable[..., np.ndarray] = dataclasses.field(init=False)

    #: A vectorized version of entity_id_to_label; initialized automatically
    _vectorized_entity_labeler: Callable[..., np.ndarray] = dataclasses.field(init=False)

    #: A vectorized version of relation_id_to_label; initialized automatically
    _vectorized_relation_labeler: Callable[..., np.ndarray] = dataclasses.field(init=False)

    def __post_init__(self):
        """Pre-compute derived mappings."""
        # ID to label mapping
        self.entity_id_to_label = invert_mapping(mapping=self.entity_to_id)
        self.relation_id_to_label = invert_mapping(mapping=self.relation_to_id)

        # vectorized versions
        self._vectorized_entity_mapper = np.vectorize(self.entity_to_id.get)
        self._vectorized_relation_mapper = np.vectorize(self.relation_to_id.get)
        self._vectorized_entity_labeler = np.vectorize(self.entity_id_to_label.get)
        self._vectorized_relation_labeler = np.vectorize(self.relation_id_to_label.get)

    @classmethod
    def from_labeled_triples(
        cls,
        triples: LabeledTriples,
        create_inverse_triples: bool = False,
        entity_to_id: Optional[EntityMapping] = None,
        relation_to_id: Optional[RelationMapping] = None,
        compact_id: bool = True,
        filter_out_candidate_inverse_relations: bool = True,
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
        :param filter_out_candidate_inverse_relations:
            Whether to remove triples with relations with the inverse suffix.

        :return:
            A new triples factory.
        """
        # Check if the triples are inverted already
        # We re-create them pure index based to ensure that _all_ inverse triples are present and that they are
        # contained if and only if create_inverse_triples is True.
        if filter_out_candidate_inverse_relations:
            unique_relations, inverse = np.unique(triples[:, 1], return_inverse=True)
            suspected_to_be_inverse_relations = {r for r in unique_relations if r.endswith(INVERSE_SUFFIX)}
            if len(suspected_to_be_inverse_relations) > 0:
                logger.warning(
                    f'Some triples already have the inverse relation suffix {INVERSE_SUFFIX}. '
                    f'Re-creating inverse triples to ensure consistency. You may disable this behaviour by passing '
                    f'filter_out_candidate_inverse_relations=False',
                )
                relation_ids_to_remove = [
                    i
                    for i, r in enumerate(unique_relations.tolist())
                    if r in suspected_to_be_inverse_relations
                ]
                mask = np.isin(element=inverse, test_elements=relation_ids_to_remove, invert=True)
                logger.info(f"Keeping {mask.sum() / mask.shape[0]} triples.")
                triples = triples[mask]

        # Generate entity mapping if necessary
        if entity_to_id is None:
            entity_to_id = create_entity_mapping(triples=triples)
        if compact_id:
            entity_to_id = compact_mapping(mapping=entity_to_id)[0]

        # Generate relation mapping if necessary
        if relation_to_id is None:
            relation_to_id = create_relation_mapping(triples[:, 1])
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
            mapped_triples=mapped_triples,
            create_inverse_triples=create_inverse_triples,
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

    def clone_and_exchange_triples(
        self,
        mapped_triples: MappedTriples,
    ) -> "TriplesFactory":
        """
        Create a new triples factory sharing everything except the triples.

        .. note ::
            We use shallow copies.

        :param mapped_triples:
            The new mapped triples.

        :return:
            The new factory.
        """
        return TriplesFactory(
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            mapped_triples=mapped_triples,
            create_inverse_triples=self.create_inverse_triples,
        )

    @property
    def num_entities(self) -> int:  # noqa: D401
        """The number of unique entities."""
        return len(self.entity_to_id)

    @property
    def num_relations(self) -> int:  # noqa: D401
        """The number of unique relations."""
        if self.create_inverse_triples:
            return 2 * self.real_num_relations
        return self.real_num_relations

    @property
    def real_num_relations(self) -> int:  # noqa: D401
        """The number of relations without inverse relations."""
        return len(self.relation_to_id)

    @property
    def num_triples(self) -> int:  # noqa: D401
        """The number of triples."""
        return self.mapped_triples.shape[0]

    @property
    def triples(self) -> np.ndarray:  # noqa: D401
        """The labeled triples, a 3-column matrix where each row are the head label, relation label, then tail label."""
        logger.warning("Reconstructing all label-based triples. This is expensive and rarely needed.")
        return self.label_triples(self.mapped_triples)

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

    def get_inverse_relation_id(self, relation: Union[str, int]) -> int:
        """Get the inverse relation identifier for the given relation."""
        if not self.create_inverse_triples:
            raise ValueError('Can not get inverse triple, they have not been created.')
        relation = next(iter(self.relations_to_ids(relations=[relation])))
        return self._get_inverse_relation_id(relation)

    @staticmethod
    def _get_inverse_relation_id(relation_id: Union[int, torch.LongTensor]) -> Union[int, torch.LongTensor]:
        return relation_id + 1

    def _add_inverse_triples_if_necessary(self, mapped_triples: MappedTriples) -> MappedTriples:
        """Add inverse triples if they shall be created."""
        if self.create_inverse_triples:
            logger.info("Creating inverse triples.")
            h, r, t = mapped_triples.t()
            mapped_triples = torch.cat([
                torch.stack([h, 2 * r, t], dim=-1),
                torch.stack([t, self._get_inverse_relation_id(2 * r), h], dim=-1),
            ])
        return mapped_triples

    def create_slcwa_instances(self) -> Instances:
        """Create sLCWA instances for this factory's triples."""
        return SLCWAInstances(mapped_triples=self._add_inverse_triples_if_necessary(mapped_triples=self.mapped_triples))

    def create_lcwa_instances(self, use_tqdm: Optional[bool] = None) -> Instances:
        """Create LCWA instances for this factory's triples."""
        return LCWAInstances.from_triples(
            mapped_triples=self._add_inverse_triples_if_necessary(mapped_triples=self.mapped_triples),
            num_entities=self.num_entities,
        )

    def label_triples(
        self,
        triples: MappedTriples,
        unknown_entity_label: str = "[UNKNOWN]",
        unknown_relation_label: Optional[str] = None,
    ) -> LabeledTriples:
        """
        Convert ID-based triples to label-based ones.

        :param triples:
            The ID-based triples.
        :param unknown_entity_label:
            The label to use for unknown entity IDs.
        :param unknown_relation_label:
            The label to use for unknown relation IDs.

        :return:
            The same triples, but labeled.
        """
        if len(triples) == 0:
            return np.empty(shape=(0, 3), dtype=str)
        if unknown_relation_label is None:
            unknown_relation_label = unknown_entity_label
        return np.stack([
            labeler(column, unknown_label)
            for (labeler, unknown_label), column in zip(
                [
                    (self._vectorized_entity_labeler, unknown_entity_label),
                    (self._vectorized_relation_labeler, unknown_relation_label),
                    (self._vectorized_entity_labeler, unknown_entity_label),
                ],
                triples.t().numpy(),
            )
        ], axis=1)

    def split(
        self,
        ratios: Union[float, Sequence[float]] = 0.8,
        *,
        random_state: TorchRandomHint = None,
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
        # input normalization
        ratios = normalize_ratios(ratios)
        generator = ensure_torch_random_state(random_state)

        # convert to absolute sizes
        sizes = get_absolute_split_sizes(n_total=self.num_triples, ratios=ratios)

        # Split indices
        idx = torch.randperm(self.num_triples, generator=generator)
        idx_groups = idx.split(split_size=sizes, dim=0)

        # Split triples
        triples_groups = [
            self.mapped_triples[idx]
            for idx in idx_groups
        ]
        logger.info(
            'done splitting triples to groups of sizes %s',
            [triples.shape[0] for triples in triples_groups],
        )

        # Make sure that the first element has all the right stuff in it
        logger.debug('cleaning up groups')
        triples_groups = _tf_cleanup_all(triples_groups, random_state=generator if randomize_cleanup else None)
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
            self.clone_and_exchange_triples(mapped_triples=triples)
            for triples in triples_groups
        ]

    def get_most_frequent_relations(self, n: Union[int, float]) -> Set[int]:
        """Get the IDs of the n most frequent relations.

        :param n: Either the (integer) number of top relations to keep or the (float) percentage of top relationships
         to keep
        """
        logger.info(f'applying cutoff of {n} to {self}')
        if isinstance(n, float):
            assert 0 < n < 1
            n = int(self.num_relations * n)
        elif not isinstance(n, int):
            raise TypeError('n must be either an integer or a float')

        uniq, counts = self.mapped_triples[:, 1].unique(return_counts=True)
        top_counts, top_ids = counts.topk(k=n, largest=True)
        return set(uniq[top_ids].tolist())

    def entities_to_ids(self, entities: Union[Collection[int], Collection[str]]) -> Collection[int]:
        """Normalize entities to IDs."""
        return _ensure_ids(labels_or_ids=entities, label_to_id=self.entity_to_id)

    def get_mask_for_entities(
        self,
        entities: Union[Collection[int], Collection[str]],
        invert: bool = False,
    ) -> torch.BoolTensor:
        """Get a boolean mask for triples with the given entities."""
        entities = self.entities_to_ids(entities=entities)
        return _get_triple_mask(
            ids=entities,
            triples=self.mapped_triples,
            columns=(0, 2),  # head and entity need to fulfil the requirement
            invert=invert,
            max_id=self.num_entities,
        )

    def relations_to_ids(
        self,
        relations: Union[Collection[int], Collection[str]],
    ) -> Collection[int]:
        """Normalize relations to IDs."""
        return _ensure_ids(labels_or_ids=relations, label_to_id=self.relation_to_id)

    def get_mask_for_relations(
        self,
        relations: Union[Collection[int], Collection[str]],
        invert: bool = False,
    ) -> torch.BoolTensor:
        """Get a boolean mask for triples with the given relations."""
        return _get_triple_mask(
            ids=self.relations_to_ids(relations=relations),
            triples=self.mapped_triples,
            columns=1,
            invert=invert,
            max_id=self.num_relations,
        )

    def entity_word_cloud(self, top: Optional[int] = None):
        """Make a word cloud based on the frequency of occurrence of each entity in a Jupyter notebook.

        :param top: The number of top entities to show. Defaults to 100.

        .. warning::

            This function requires the ``word_cloud`` package. Use ``pip install pykeen[plotting]`` to
            install it automatically, or install it yourself with
            ``pip install git+https://github.com/kavgan/word_cloud.git``.
        """
        return self._word_cloud(ids=self.mapped_triples[:, [0, 2]], id_to_label=self.entity_id_to_label, top=top or 100)

    def relation_word_cloud(self, top: Optional[int] = None):
        """Make a word cloud based on the frequency of occurrence of each relation in a Jupyter notebook.

        :param top: The number of top relations to show. Defaults to 100.

        .. warning::

            This function requires the ``word_cloud`` package. Use ``pip install pykeen[plotting]`` to
            install it automatically, or install it yourself with
            ``pip install git+https://github.com/kavgan/word_cloud.git``.
        """
        return self._word_cloud(ids=self.mapped_triples[:, 1], id_to_label=self.relation_id_to_label, top=top or 100)

    def _word_cloud(self, *, ids: torch.LongTensor, id_to_label: Mapping[int, str], top: int):
        try:
            from word_cloud.word_cloud_generator import WordCloud
        except ImportError:
            logger.warning(
                'Could not import module `word_cloud`. '
                'Try installing it with `pip install git+https://github.com/kavgan/word_cloud.git`',
            )
            return

        # pre-filter to keep only topk
        uniq, counts = ids.view(-1).unique(return_counts=True)
        top_counts, top_ids = counts.topk(k=top, largest=True)

        # generate text
        text = list(itertools.chain(*(
            itertools.repeat(id_to_label[e_id], count)
            for e_id, count in zip(top_ids.tolist(), top_counts.tolist())
        )))

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
        for column, id_to_label in dict(
            head=self._vectorized_entity_labeler,
            relation=self._vectorized_relation_labeler,
            tail=self._vectorized_entity_labeler,
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
        entities: Union[None, Collection[int], Collection[str]] = None,
        relations: Union[None, Collection[int], Collection[str]] = None,
        invert_entity_selection: bool = False,
        invert_relation_selection: bool = False,
    ) -> 'TriplesFactory':
        """Make a new triples factory only keeping the given entities and relations, but keeping the ID mapping.

        :param entities:
            The entities of interest. If None, defaults to all entities.
        :param relations:
            The relations of interest. If None, defaults to all relations.
        :param invert_entity_selection:
            Whether to invert the entity selection, i.e. select those triples without the provided entities.
        :param invert_relation_selection:
            Whether to invert the relation selection, i.e. select those triples without the provided relations.

        :return:
            A new triples factory, which has only a subset of the triples containing the entities and relations of
            interest. The label-to-ID mapping is *not* modified.
        """
        keep_mask = None

        # Filter for entities
        if entities is not None:
            keep_mask = self.get_mask_for_entities(entities=entities, invert=invert_entity_selection)
            remaining_entities = self.num_entities - len(entities) if invert_entity_selection else len(entities)
            logger.info(
                f"Keeping {remaining_entities}/{self.num_entities} "
                f"({remaining_entities / self.num_entities:2.2%}) entities."
            )

        # Filter for relations
        if relations is not None:
            relation_mask = self.get_mask_for_relations(relations=relations, invert=invert_relation_selection)
            remaining_relations = self.num_relations - len(relations) if invert_entity_selection else len(relations)
            logger.info(
                f"Keeping {remaining_relations}/{self.num_relations} "
                f"({remaining_relations / self.num_relations:2.2%}) relations."
            )
            keep_mask = relation_mask if keep_mask is None else keep_mask & relation_mask

        # No filtering happened
        if keep_mask is None:
            return self

        num_triples = keep_mask.sum()
        logger.info(f"Keeping {num_triples}/{self.num_triples} ({num_triples / self.num_triples:2.2%}) triples.")
        return self.clone_and_exchange_triples(mapped_triples=self.mapped_triples[keep_mask])


def _tf_cleanup_all(
    triples_groups: List[MappedTriples],
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
        max_ids = [
            max(training[:, col].max().item(), testing[:, col].max().item()) + 1
            for col in columns
        ]
    for col, max_id in zip(columns, max_ids):
        # IDs not in training
        not_in_training_mask = torch.ones(max_id, dtype=torch.bool)
        not_in_training_mask[training[:, col].view(-1)] = False

        # triples with exclusive test IDs
        exclusive_triples = not_in_training_mask[testing[:, col].view(-1)].view(-1, len(col)).any(dim=-1)
        to_move_mask = to_move_mask | exclusive_triples
    return to_move_mask
