# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

import dataclasses
import itertools
import logging
import pathlib
import re
import warnings
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    TextIO,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .instances import BatchedSLCWAInstances, LCWAInstances, SubGraphSLCWAInstances
from .splitting import split
from .utils import TRIPLES_DF_COLUMNS, get_entities, get_relations, load_triples, tensor_to_df
from ..typing import (
    LABEL_HEAD,
    LABEL_RELATION,
    LABEL_TAIL,
    EntityMapping,
    LabeledTriples,
    MappedTriples,
    RelationMapping,
    TorchRandomHint,
)
from ..utils import compact_mapping, format_relative_comparison, invert_mapping, triple_tensor_to_set

__all__ = [
    "CoreTriplesFactory",
    "TriplesFactory",
    "create_entity_mapping",
    "create_relation_mapping",
    "INVERSE_SUFFIX",
    "cat_triples",
    "splits_steps",
    "splits_similarity",
    "RelationInverter",
    "relation_inverter",
]

logger = logging.getLogger(__name__)

INVERSE_SUFFIX = "_inverse"


def create_entity_mapping(triples: LabeledTriples) -> EntityMapping:
    """Create mapping from entity labels to IDs.

    :param triples: shape: (n, 3), dtype: str
    """
    # Split triples
    heads, tails = triples[:, 0], triples[:, 2]
    # Sorting ensures consistent results when the triples are permuted
    entity_labels = sorted(set(heads).union(tails))
    # Create mapping
    return {str(label): i for (i, label) in enumerate(entity_labels)}


def create_relation_mapping(relations: set) -> RelationMapping:
    """Create mapping from relation labels to IDs.

    :param relations: set
    """
    # Sorting ensures consistent results when the triples are permuted
    relation_labels = sorted(
        set(relations),
        key=lambda x: (re.sub(f"{INVERSE_SUFFIX}$", "", x), x.endswith(f"{INVERSE_SUFFIX}")),
    )
    # Create mapping
    return {str(label): i for (i, label) in enumerate(relation_labels)}


def _map_triples_elements_to_ids(
    triples: LabeledTriples,
    entity_to_id: EntityMapping,
    relation_to_id: RelationMapping,
) -> MappedTriples:
    """Map entities and relations to pre-defined ids."""
    if triples.size == 0:
        logger.warning("Provided empty triples to map.")
        return torch.empty(0, 3, dtype=torch.long)

    # When triples that don't exist are trying to be mapped, they get the id "-1"
    entity_getter = np.vectorize(entity_to_id.get)
    head_column = entity_getter(triples[:, 0:1], [-1])
    tail_column = entity_getter(triples[:, 2:3], [-1])
    relation_getter = np.vectorize(relation_to_id.get)
    relation_column = relation_getter(triples[:, 1:2], [-1])

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
        non_mappable_triples = head_filter | relation_filter | tail_filter
        head_column = head_column[~non_mappable_triples, None]
        relation_column = relation_column[~non_mappable_triples, None]
        tail_column = tail_column[~non_mappable_triples, None]
        logger.warning(
            f"In total {non_mappable_triples.sum():.0f} from {triples.shape[0]:.0f} triples were filtered out",
        )

    triples_of_ids = np.concatenate([head_column, relation_column, tail_column], axis=1)

    triples_of_ids = np.array(triples_of_ids, dtype=np.int64)
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
    mask = torch.isin(
        elements=triples,
        test_elements=torch.as_tensor(list(ids), dtype=torch.long),
        assume_unique=False,
        invert=invert,
    )
    if len(columns) > 1:
        mask = mask.all(dim=-1)
    return mask


def _ensure_ids(
    labels_or_ids: Union[Collection[int], Collection[str]],
    label_to_id: Mapping[str, int],
) -> Collection[int]:
    """Convert labels to IDs."""
    return [label_to_id[l_or_i] if isinstance(l_or_i, str) else l_or_i for l_or_i in labels_or_ids]


RelationID = TypeVar("RelationID", int, torch.LongTensor)


class RelationInverter:
    """An interface for inverse-relation ID mapping."""

    # TODO: method is_inverse?

    @abstractmethod
    def get_inverse_id(self, relation_id: RelationID) -> RelationID:
        """Get the inverse ID for a given relation."""
        # TODO: inverse of inverse?
        raise NotImplementedError

    @abstractmethod
    def _map(self, batch: torch.LongTensor, index: int = 1) -> torch.LongTensor:
        raise NotImplementedError

    @abstractmethod
    def invert_(self, batch: torch.LongTensor, index: int = 1) -> torch.LongTensor:
        """Invert relations in a batch (in-place)."""
        raise NotImplementedError

    def map(self, batch: torch.LongTensor, index: int = 1, invert: bool = False) -> torch.LongTensor:
        """Map relations of batch, optionally also inverting them."""
        batch = self._map(batch=batch, index=index)
        return self.invert_(batch=batch, index=index) if invert else batch


class DefaultRelationInverter(RelationInverter):
    """Maps normal relations to even IDs, and the corresponding inverse to the next odd ID."""

    def get_inverse_id(self, relation_id: RelationID) -> RelationID:  # noqa: D102
        return relation_id + 1

    def _map(self, batch: torch.LongTensor, index: int = 1, invert: bool = False) -> torch.LongTensor:  # noqa: D102
        batch = batch.clone()
        batch[:, index] *= 2
        return batch

    def invert_(self, batch: torch.LongTensor, index: int = 1) -> torch.LongTensor:  # noqa: D102
        # The number of relations stored in the triples factory includes the number of inverse relations
        # Id of inverse relation: relation + 1
        batch[:, index] += 1
        return batch


relation_inverter = DefaultRelationInverter()


@dataclasses.dataclass
class Labeling:
    """A mapping between labels and IDs."""

    #: The mapping from labels to IDs.
    label_to_id: Mapping[str, int]

    #: The inverse mapping for label_to_id; initialized automatically
    id_to_label: Mapping[int, str] = dataclasses.field(init=False)

    #: A vectorized version of entity_label_to_id; initialized automatically
    _vectorized_mapper: Callable[..., np.ndarray] = dataclasses.field(init=False, compare=False)

    #: A vectorized version of entity_id_to_label; initialized automatically
    _vectorized_labeler: Callable[..., np.ndarray] = dataclasses.field(init=False, compare=False)

    def __post_init__(self):
        """Precompute inverse mappings."""
        self.id_to_label = invert_mapping(mapping=self.label_to_id)
        self._vectorized_mapper = np.vectorize(self.label_to_id.get, otypes=[int])
        self._vectorized_labeler = np.vectorize(self.id_to_label.get, otypes=[str])

    def label(
        self,
        ids: Union[int, Sequence[int], np.ndarray, torch.LongTensor],
        unknown_label: str = "unknown",
    ) -> np.ndarray:
        """Convert IDs to labels."""
        # Normalize input
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy()
        if isinstance(ids, int):
            ids = [ids]
        ids = np.asanyarray(ids)
        # label
        return self._vectorized_labeler(ids, (unknown_label,))


def restrict_triples(
    mapped_triples: MappedTriples,
    entities: Optional[Collection[int]] = None,
    relations: Optional[Collection[int]] = None,
    invert_entity_selection: bool = False,
    invert_relation_selection: bool = False,
) -> MappedTriples:
    """Select a subset of triples based on the given entity and relation ID selection.

    :param mapped_triples:
        The ID-based triples.
    :param entities:
        The entity IDs of interest. If None, defaults to all entities.
    :param relations:
        The relation IDs of interest. If None, defaults to all relations.
    :param invert_entity_selection:
        Whether to invert the entity selection, i.e. select those triples without the provided entities.
    :param invert_relation_selection:
        Whether to invert the relation selection, i.e. select those triples without the provided relations.
    :return:
        A tensor of triples containing the entities and relations of interest.
    """
    keep_mask = None

    # Filter for entities
    if entities is not None:
        keep_mask = _get_triple_mask(
            ids=entities,
            triples=mapped_triples,
            columns=(0, 2),  # head and entity need to fulfil the requirement
            invert=invert_entity_selection,
        )

    # Filter for relations
    if relations is not None:
        relation_mask = _get_triple_mask(
            ids=relations,
            triples=mapped_triples,
            columns=1,
            invert=invert_relation_selection,
        )
        keep_mask = relation_mask if keep_mask is None else keep_mask & relation_mask

    # No filter
    if keep_mask is None:
        return mapped_triples

    return mapped_triples[keep_mask]


@dataclasses.dataclass
class CoreTriplesFactory:
    """Create instances from ID-based triples."""

    triples_file_name: ClassVar[str] = "numeric_triples.tsv.gz"
    base_file_name: ClassVar[str] = "base.pth"

    def __init__(
        self,
        mapped_triples: MappedTriples,
        num_entities: int,
        num_relations: int,
        entity_ids: Collection[int],
        relation_ids: Collection[int],
        create_inverse_triples: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        """
        Create the triples factory.

        :param mapped_triples: shape: (n, 3)
            A three-column matrix where each row are the head identifier, relation identifier, then tail identifier.
        :param num_entities:
            The number of entities.
        :param num_relations:
            The number of relations.
        :param create_inverse_triples:
            Whether to create inverse triples.
        :param metadata:
            Arbitrary metadata to go with the graph
        """
        super().__init__()
        self.mapped_triples = mapped_triples
        self._num_entities = num_entities
        self._num_relations = num_relations
        self.entity_ids = entity_ids
        self.relation_ids = relation_ids
        self.create_inverse_triples = create_inverse_triples
        if metadata is None:
            metadata = dict()
        self.metadata = metadata

    @classmethod
    def create(
        cls,
        mapped_triples: MappedTriples,
        num_entities: Optional[int] = None,
        num_relations: Optional[int] = None,
        entity_ids: Collection[int] = None,
        relation_ids: Collection[int] = None,
        create_inverse_triples: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "CoreTriplesFactory":
        """
        Create a triples factory without any label information.

        :param mapped_triples: shape: (n, 3)
            The ID-based triples.
        :param num_entities:
            The number of entities. If not given, inferred from mapped_triples.
        :param num_relations:
            The number of relations. If not given, inferred from mapped_triples.
        :param create_inverse_triples:
            Whether to create inverse triples.
        :param metadata:
            Additional metadata to store in the factory.

        :return:
            A new triples factory.
        """
        if num_entities is None:
            num_entities = mapped_triples[:, [0, 2]].max().item() + 1
        if num_relations is None:
            num_relations = mapped_triples[:, 1].max().item() + 1
        if entity_ids is None:
            entity_ids = get_entities(mapped_triples)
        if relation_ids is None:
            relation_ids = get_relations(mapped_triples)
        return CoreTriplesFactory(
            mapped_triples=mapped_triples,
            num_entities=num_entities,
            num_relations=num_relations,
            entity_ids=entity_ids,
            relation_ids=relation_ids,
            create_inverse_triples=create_inverse_triples,
            metadata=metadata,
        )

    def __eq__(self, __o: object) -> bool:  # noqa: D105
        if not isinstance(__o, CoreTriplesFactory):
            return False
        return (
            (self.num_entities == __o.num_entities)
            and (self.num_relations == __o.num_relations)
            and (self.num_triples == __o.num_triples)
            and (self.create_inverse_triples == __o.create_inverse_triples)
            and bool((self.mapped_triples == __o.mapped_triples).all().item())
        )

    @property
    def num_entities(self) -> int:  # noqa: D401
        """The number of unique entities."""
        return self._num_entities

    @property
    def num_relations(self) -> int:  # noqa: D401
        """The number of unique relations."""
        if self.create_inverse_triples:
            return 2 * self.real_num_relations
        return self.real_num_relations

    @property
    def real_num_relations(self) -> int:  # noqa: D401
        """The number of relations without inverse relations."""
        return self._num_relations

    @property
    def num_triples(self) -> int:  # noqa: D401
        """The number of triples."""
        return self.mapped_triples.shape[0]

    def extra_repr(self) -> str:
        """Extra representation string."""
        d = [
            ("num_entities", self.num_entities),
            ("num_relations", self.num_relations),
            ("num_triples", self.num_triples),
            ("inverse_triples", self.create_inverse_triples),
        ]
        d.extend(sorted(self.metadata.items()))  # type: ignore
        return ", ".join(f'{k}="{v}"' if isinstance(v, (str, pathlib.Path)) else f"{k}={v}" for k, v in d)

    def __repr__(self):  # noqa: D105
        return f"{self.__class__.__name__}({self.extra_repr()})"

    def with_labels(
        self,
        entity_to_id: Mapping[str, int],
        relation_to_id: Mapping[str, int],
    ) -> "TriplesFactory":
        """Add labeling to the TriplesFactory."""
        # check new label to ID mappings
        for name, columns, new_labeling in (
            ("entity", [0, 2], entity_to_id),
            ("relation", 1, relation_to_id),
        ):
            existing_ids = set(self.mapped_triples[:, columns].unique().tolist())
            if not existing_ids.issubset(new_labeling.values()):
                diff = existing_ids.difference(new_labeling.values())
                raise ValueError(f"Some existing IDs do not occur in the new {name} labeling: {diff}")
        return TriplesFactory(
            mapped_triples=self.mapped_triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            create_inverse_triples=self.create_inverse_triples,
            metadata=self.metadata,
        )

    def get_inverse_relation_id(self, relation: int) -> int:
        """Get the inverse relation identifier for the given relation."""
        if not self.create_inverse_triples:
            raise ValueError("Can not get inverse triple, they have not been created.")
        return relation_inverter.get_inverse_id(relation_id=relation)

    def _add_inverse_triples_if_necessary(self, mapped_triples: MappedTriples) -> MappedTriples:
        """Add inverse triples if they shall be created."""
        if not self.create_inverse_triples:
            return mapped_triples

        logger.info("Creating inverse triples.")
        return torch.cat(
            [
                relation_inverter.map(batch=mapped_triples),
                relation_inverter.map(batch=mapped_triples, invert=True).flip(1),
            ]
        )

    def create_slcwa_instances(self, *, sampler: Optional[str] = None, **kwargs) -> Dataset:
        """Create sLCWA instances for this factory's triples."""
        cls = BatchedSLCWAInstances if sampler is None else SubGraphSLCWAInstances
        if "shuffle" in kwargs:
            if kwargs.pop("shuffle"):
                warnings.warn("Training instances are always shuffled.", DeprecationWarning)
            else:
                raise AssertionError("If shuffle is provided, it must be True.")
        return cls(
            mapped_triples=self._add_inverse_triples_if_necessary(mapped_triples=self.mapped_triples),
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            **kwargs,
        )

    def create_lcwa_instances(self, use_tqdm: Optional[bool] = None, target: Optional[int] = None) -> Dataset:
        """Create LCWA instances for this factory's triples."""
        return LCWAInstances.from_triples(
            mapped_triples=self._add_inverse_triples_if_necessary(mapped_triples=self.mapped_triples),
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            target=target,
        )

    def get_most_frequent_relations(self, n: Union[int, float]) -> Set[int]:
        """Get the IDs of the n most frequent relations.

        :param n:
            Either the (integer) number of top relations to keep or the (float) percentage of top relationships to keep.
        """
        logger.info(f"applying cutoff of {n} to {self}")
        if isinstance(n, float):
            assert 0 < n < 1
            n = int(self.num_relations * n)
        elif not isinstance(n, int):
            raise TypeError("n must be either an integer or a float")

        uniq, counts = self.mapped_triples[:, 1].unique(return_counts=True)
        top_counts, top_ids = counts.topk(k=n, largest=True)
        return set(uniq[top_ids].tolist())

    def clone_and_exchange_triples(
        self,
        mapped_triples: MappedTriples,
        extra_metadata: Optional[Dict[str, Any]] = None,
        keep_metadata: bool = True,
        create_inverse_triples: Optional[bool] = None,
    ) -> "CoreTriplesFactory":
        """
        Create a new triples factory sharing everything except the triples.

        .. note ::
            We use shallow copies.

        :param mapped_triples:
            The new mapped triples.
        :param extra_metadata:
            Extra metadata to include in the new triples factory. If ``keep_metadata`` is true,
            the dictionaries will be unioned with precedence taken on keys from ``extra_metadata``.
        :param keep_metadata:
            Pass the current factory's metadata to the new triples factory
        :param create_inverse_triples:
            Change inverse triple creation flag. If None, use flag from this factory.

        :return:
            The new factory.
        """
        if create_inverse_triples is None:
            create_inverse_triples = self.create_inverse_triples
        return CoreTriplesFactory(
            mapped_triples=mapped_triples,
            num_entities=self.num_entities,
            num_relations=self.real_num_relations,
            entity_ids=self.entity_ids,
            relation_ids=self.relation_ids,
            create_inverse_triples=create_inverse_triples,
            metadata={
                **(extra_metadata or {}),
                **(self.metadata if keep_metadata else {}),  # type: ignore
            },
        )

    def split(
        self,
        ratios: Union[float, Sequence[float]] = 0.8,
        *,
        random_state: TorchRandomHint = None,
        randomize_cleanup: bool = False,
        method: Optional[str] = None,
    ) -> List["CoreTriplesFactory"]:
        """Split a triples factory into a train/test.

        :param ratios:
            There are three options for this argument:

            1. A float can be given between 0 and 1.0, non-inclusive. The first set of triples will
               get this ratio and the second will get the rest.
            2. A list of ratios can be given for which set in which order should get what ratios as in
               ``[0.8, 0.1]``. The final ratio can be omitted because that can be calculated.
            3. All ratios can be explicitly set in order such as in ``[0.8, 0.1, 0.1]``
               where the sum of all ratios is 1.0.
        :param random_state:
            The random state used to shuffle and split the triples.
        :param randomize_cleanup:
            If true, uses the non-deterministic method for moving triples to the training set. This has the
            advantage that it does not necessarily have to move all of them, but it might be significantly
            slower since it moves one triple at a time.
        :param method:
            The name of the method to use, from SPLIT_METHODS. Defaults to "coverage".

        :return:
            A partition of triples, which are split (approximately) according to the ratios, stored TriplesFactory's
            which share everything else with this root triples factory.

        .. code-block:: python

            ratio = 0.8  # makes a [0.8, 0.2] split
            training_factory, testing_factory = factory.split(ratio)

            ratios = [0.8, 0.1]  # makes a [0.8, 0.1, 0.1] split
            training_factory, testing_factory, validation_factory = factory.split(ratios)

            ratios = [0.8, 0.1, 0.1]  # also makes a [0.8, 0.1, 0.1] split
            training_factory, testing_factory, validation_factory = factory.split(ratios)
        """
        # Make new triples factories for each group
        return [
            self.clone_and_exchange_triples(
                mapped_triples=triples,
                # do not explicitly create inverse triples for testing; this is handled by the evaluation code
                create_inverse_triples=None if i == 0 else False,
            )
            for i, triples in enumerate(
                split(
                    mapped_triples=self.mapped_triples,
                    ratios=ratios,
                    random_state=random_state,
                    randomize_cleanup=randomize_cleanup,
                    method=method,
                )
            )
        ]

    def entities_to_ids(self, entities: Union[Collection[int], Collection[str]]) -> Collection[int]:
        """Normalize entities to IDs."""
        for e in entities:
            if not isinstance(e, int):
                raise ValueError(f"{self.__class__.__name__} cannot convert entity IDs from {type(e)} to int.")
        return cast(Collection[int], entities)

    def relations_to_ids(self, relations: Union[Collection[int], Collection[str]]) -> Collection[int]:
        """Normalize relations to IDs."""
        for e in relations:
            if not isinstance(e, int):
                raise ValueError(f"{self.__class__.__name__} cannot convert relation IDs from {type(e)} to int.")
        return cast(Collection[int], relations)

    def get_mask_for_relations(
        self,
        relations: Collection[int],
        invert: bool = False,
    ) -> torch.BoolTensor:
        """Get a boolean mask for triples with the given relations."""
        return _get_triple_mask(
            ids=relations,
            triples=self.mapped_triples,
            columns=1,
            invert=invert,
            max_id=self.num_relations,
        )

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
        return tensor_to_df(tensor=tensor, **kwargs)

    def new_with_restriction(
        self,
        entities: Union[None, Collection[int], Collection[str]] = None,
        relations: Union[None, Collection[int], Collection[str]] = None,
        invert_entity_selection: bool = False,
        invert_relation_selection: bool = False,
    ) -> "CoreTriplesFactory":
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
        # prepare metadata
        extra_metadata = {}
        if entities is not None:
            extra_metadata["entity_restriction"] = entities
            entities = self.entities_to_ids(entities=entities)
            remaining_entities = (self.num_entities - len(entities)) if invert_entity_selection else len(entities)
            logger.info(f"keeping {format_relative_comparison(remaining_entities, self.num_entities)} entities.")
        if relations is not None:
            extra_metadata["relation_restriction"] = relations
            relations = self.relations_to_ids(relations=relations)
            remaining_relations = (self.num_relations - len(relations)) if invert_relation_selection else len(relations)
            logger.info(f"keeping {format_relative_comparison(remaining_relations, self.num_relations)} relations.")

        # Delegate to function
        mapped_triples = restrict_triples(
            mapped_triples=self.mapped_triples,
            entities=entities,
            relations=relations,
            invert_entity_selection=invert_entity_selection,
            invert_relation_selection=invert_relation_selection,
        )

        # restrict triples can only remove triples; thus, if the new size equals the old one, nothing has changed
        if mapped_triples.shape[0] == self.num_triples:
            return self

        logger.info(f"keeping {format_relative_comparison(mapped_triples.shape[0], self.num_triples)} triples.")

        return self.clone_and_exchange_triples(
            mapped_triples=mapped_triples,
            extra_metadata=extra_metadata,
        )

    @classmethod
    def from_path_binary(
        cls,
        path: Union[str, pathlib.Path, TextIO],
    ) -> "CoreTriplesFactory":  # noqa: D102
        """
        Load triples factory from a binary file.

        :param path:
            The path, pointing to an existing PyTorch .pt file.

        :return:
            The loaded triples factory.
        """
        path = normalize_path(path)
        logger.info(f"Loading from {path.as_uri()}")
        return cls(**cls._from_path_binary(path=path))

    @classmethod
    def _from_path_binary(
        cls,
        path: pathlib.Path,
    ) -> MutableMapping[str, Any]:
        # load base
        data = dict(torch.load(path.joinpath(cls.base_file_name)))
        # load numeric triples
        data["mapped_triples"] = torch.as_tensor(
            pd.read_csv(path.joinpath(cls.triples_file_name), sep="\t", dtype=int).values,
            dtype=torch.long,
        )
        return data

    def to_path_binary(
        self,
        path: Union[str, pathlib.Path, TextIO],
    ) -> pathlib.Path:
        """
        Save triples factory to path in (PyTorch's .pt) binary format.

        :param path:
            The path to store the triples factory to.
        """
        path = normalize_path(path)
        path.mkdir(exist_ok=True, parents=True)

        # store numeric triples
        pd.DataFrame(
            data=self.mapped_triples.numpy(),
            columns=[LABEL_HEAD, LABEL_RELATION, LABEL_TAIL],
        ).to_csv(path.joinpath(self.triples_file_name), sep="\t", index=False)

        # store metadata
        torch.save(self._get_binary_state(), path.joinpath(self.base_file_name))
        logger.info(f"Stored {self} to {path.as_uri()}")

        return path

    def _get_binary_state(self):
        return dict(
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            entity_ids=self.entity_ids,
            relation_ids=self.relation_ids,
            create_inverse_triples=self.create_inverse_triples,
            metadata=self.metadata,
        )


class TriplesFactory(CoreTriplesFactory):
    """Create instances given the path to triples."""

    file_name_entity_to_id: ClassVar[str] = "entity_to_id"
    file_name_relation_to_id: ClassVar[str] = "relation_to_id"

    def __init__(
        self,
        mapped_triples: MappedTriples,
        entity_to_id: EntityMapping,
        relation_to_id: RelationMapping,
        create_inverse_triples: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        """
        Create the triples factory.

        :param mapped_triples: shape: (n, 3)
            A three-column matrix where each row are the head identifier, relation identifier, then tail identifier.
        :param entity_to_id:
            The mapping from entities' labels to their indices.
        :param relation_to_id:
            The mapping from relations' labels to their indices.
        :param create_inverse_triples:
            Whether to create inverse triples.
        :param metadata:
            Arbitrary metadata to go with the graph
        """
        super().__init__(
            mapped_triples=mapped_triples,
            num_entities=len(entity_to_id),
            num_relations=len(relation_to_id),
            entity_ids=sorted(entity_to_id.values()),
            relation_ids=sorted(relation_to_id.values()),
            create_inverse_triples=create_inverse_triples,
            metadata=metadata,
        )
        self.entity_labeling = Labeling(label_to_id=entity_to_id)
        self.relation_labeling = Labeling(label_to_id=relation_to_id)

    @classmethod
    def from_labeled_triples(
        cls,
        triples: LabeledTriples,
        create_inverse_triples: bool = False,
        entity_to_id: Optional[EntityMapping] = None,
        relation_to_id: Optional[RelationMapping] = None,
        compact_id: bool = True,
        filter_out_candidate_inverse_relations: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "TriplesFactory":
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
        :param metadata:
            Arbitrary key/value pairs to store as metadata

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
                    f"Some triples already have the inverse relation suffix {INVERSE_SUFFIX}. "
                    f"Re-creating inverse triples to ensure consistency. You may disable this behaviour by passing "
                    f"filter_out_candidate_inverse_relations=False",
                )
                relation_ids_to_remove = [
                    i for i, r in enumerate(unique_relations.tolist()) if r in suspected_to_be_inverse_relations
                ]
                mask = np.isin(element=inverse, test_elements=relation_ids_to_remove, invert=True)
                logger.info(f"keeping {mask.sum() / mask.shape[0]} triples.")
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
            metadata=metadata,
        )

    @classmethod
    def from_path(
        cls,
        path: Union[str, pathlib.Path, TextIO],
        create_inverse_triples: bool = False,
        entity_to_id: Optional[EntityMapping] = None,
        relation_to_id: Optional[RelationMapping] = None,
        compact_id: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        load_triples_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> "TriplesFactory":
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
        :param metadata:
            Arbitrary key/value pairs to store as metadata with the triples factory. Do not
            include ``path`` as a key because it is automatically taken from the ``path``
            kwarg to this function.
        :param load_triples_kwargs: Optional keyword arguments to pass to :func:`load_triples`.
            Could include the ``delimiter`` or a ``column_remapping``.

        :return:
            A new triples factory.
        """
        path = normalize_path(path)

        # TODO: Check if lazy evaluation would make sense
        triples = load_triples(path, **(load_triples_kwargs or {}))

        return cls.from_labeled_triples(
            triples=triples,
            create_inverse_triples=create_inverse_triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            compact_id=compact_id,
            metadata={
                "path": path,
                **(metadata or {}),
            },
        )

    def __eq__(self, __o: object) -> bool:  # noqa: D105
        return (
            isinstance(__o, TriplesFactory)
            and super().__eq__(__o)
            and (self.entity_to_id == __o.entity_to_id)
            and (self.relation_to_id == __o.relation_to_id)
        )

    def to_core_triples_factory(self) -> CoreTriplesFactory:
        """Return this factory as a core factory."""
        return CoreTriplesFactory(
            mapped_triples=self.mapped_triples,
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            entity_ids=self.entity_ids,
            relation_ids=self.relation_ids,
            create_inverse_triples=self.create_inverse_triples,
            metadata=self.metadata,
        )

    def to_path_binary(self, path: Union[str, pathlib.Path, TextIO]) -> pathlib.Path:  # noqa: D102
        path = super().to_path_binary(path=path)
        # store entity/relation to ID
        for name, data in (
            (
                self.file_name_entity_to_id,
                self.entity_to_id,
            ),
            (
                self.file_name_relation_to_id,
                self.relation_to_id,
            ),
        ):
            pd.DataFrame(data=data.items(), columns=["label", "id"],).sort_values(by="id").set_index("id").to_csv(
                path.joinpath(f"{name}.tsv.gz"),
                sep="\t",
            )
        return path

    @classmethod
    def _from_path_binary(cls, path: pathlib.Path) -> MutableMapping[str, Any]:
        data = super()._from_path_binary(path)
        # load entity/relation to ID
        for name in [cls.file_name_entity_to_id, cls.file_name_relation_to_id]:
            df = pd.read_csv(
                path.joinpath(f"{name}.tsv.gz"),
                sep="\t",
            )
            data[name] = dict(zip(df["label"], df["id"]))
        return data

    def _get_binary_state(self):  # noqa: D102
        return dict(
            create_inverse_triples=self.create_inverse_triples,
            metadata=self.metadata,
        )

    def clone_and_exchange_triples(
        self,
        mapped_triples: MappedTriples,
        extra_metadata: Optional[Dict[str, Any]] = None,
        keep_metadata: bool = True,
        create_inverse_triples: Optional[bool] = None,
    ) -> "TriplesFactory":  # noqa: D102
        if create_inverse_triples is None:
            create_inverse_triples = self.create_inverse_triples
        return TriplesFactory(
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            mapped_triples=mapped_triples,
            create_inverse_triples=create_inverse_triples,
            metadata={
                **(extra_metadata or {}),
                **(self.metadata if keep_metadata else {}),  # type: ignore
            },
        )

    @property
    def entity_to_id(self) -> Mapping[str, int]:
        """Return the mapping from entity labels to IDs."""
        return self.entity_labeling.label_to_id

    @property
    def entity_id_to_label(self) -> Mapping[int, str]:
        """Return the mapping from entity IDs to labels."""
        return self.entity_labeling.id_to_label

    @property
    def relation_to_id(self) -> Mapping[str, int]:
        """Return the mapping from relations labels to IDs."""
        return self.relation_labeling.label_to_id

    @property
    def relation_id_to_label(self) -> Mapping[int, str]:
        """Return the mapping from relations IDs to labels."""
        return self.relation_labeling.id_to_label

    @property
    def triples(self) -> np.ndarray:  # noqa: D401
        """The labeled triples, a 3-column matrix where each row are the head label, relation label, then tail label."""
        logger.warning("Reconstructing all label-based triples. This is expensive and rarely needed.")
        return self.label_triples(self.mapped_triples)

    def get_inverse_relation_id(self, relation: Union[str, int]) -> int:
        """Get the inverse relation identifier for the given relation."""
        relation = next(iter(self.relations_to_ids(relations=[relation])))  # type: ignore
        return super().get_inverse_relation_id(relation=relation)

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
        return np.stack(
            [
                labeling.label(ids=column, unknown_label=unknown_label)
                for (labeling, unknown_label), column in zip(
                    [
                        (self.entity_labeling, unknown_entity_label),
                        (self.relation_labeling, unknown_relation_label),
                        (self.entity_labeling, unknown_entity_label),
                    ],
                    triples.t().numpy(),
                )
            ],
            axis=1,
        )

    def entities_to_ids(self, entities: Union[Collection[int], Collection[str]]) -> Collection[int]:  # noqa: D102
        return _ensure_ids(labels_or_ids=entities, label_to_id=self.entity_labeling.label_to_id)

    def relations_to_ids(self, relations: Union[Collection[int], Collection[str]]) -> Collection[int]:  # noqa: D102
        return _ensure_ids(labels_or_ids=relations, label_to_id=self.relation_labeling.label_to_id)

    def get_mask_for_relations(
        self,
        relations: Union[Collection[int], Collection[str]],
        invert: bool = False,
    ) -> torch.BoolTensor:
        """Get a boolean mask for triples with the given relations."""
        return super().get_mask_for_relations(relations=self.relations_to_ids(relations=relations))

    def entity_word_cloud(self, top: Optional[int] = None):
        """Make a word cloud based on the frequency of occurrence of each entity in a Jupyter notebook.

        :param top: The number of top entities to show. Defaults to 100.

        .. warning::

            This function requires the ``word_cloud`` package. Use ``pip install pykeen[plotting]`` to
            install it automatically, or install it yourself with
            ``pip install git+https://github.com/kavgan/word_cloud.git``.
        """
        return self._word_cloud(
            ids=self.mapped_triples[:, [0, 2]],
            id_to_label=self.entity_labeling.id_to_label,
            top=top or 100,
        )

    def relation_word_cloud(self, top: Optional[int] = None):
        """Make a word cloud based on the frequency of occurrence of each relation in a Jupyter notebook.

        :param top: The number of top relations to show. Defaults to 100.

        .. warning::

            This function requires the ``word_cloud`` package. Use ``pip install pykeen[plotting]`` to
            install it automatically, or install it yourself with
            ``pip install git+https://github.com/kavgan/word_cloud.git``.
        """
        return self._word_cloud(
            ids=self.mapped_triples[:, 1],
            id_to_label=self.relation_labeling.id_to_label,
            top=top or 100,
        )

    def _word_cloud(self, *, ids: torch.LongTensor, id_to_label: Mapping[int, str], top: int):
        try:
            from word_cloud.word_cloud_generator import WordCloud
        except ImportError:
            logger.warning(
                "Could not import module `word_cloud`. "
                "Try installing it with `pip install git+https://github.com/kavgan/word_cloud.git`",
            )
            return

        # pre-filter to keep only topk
        uniq, counts = ids.view(-1).unique(return_counts=True)

        # if top is larger than the number of available options
        top = min(top, uniq.numel())
        top_counts, top_ids = counts.topk(k=top, largest=True)

        # generate text
        text = list(
            itertools.chain(
                *(
                    itertools.repeat(id_to_label[e_id], count)
                    for e_id, count in zip(top_ids.tolist(), top_counts.tolist())
                )
            )
        )

        from IPython.core.display import HTML

        word_cloud = WordCloud()
        return HTML(word_cloud.get_embed_code(text=text, topn=top))

    def tensor_to_df(
        self,
        tensor: torch.LongTensor,
        **kwargs: Union[torch.Tensor, np.ndarray, Sequence],
    ) -> pd.DataFrame:  # noqa: D102
        data = super().tensor_to_df(tensor=tensor, **kwargs)
        old_col = list(data.columns)

        # vectorized label lookup
        for column, labeling in dict(
            head=self.entity_labeling,
            relation=self.relation_labeling,
            tail=self.entity_labeling,
        ).items():
            assert labeling is not None
            data[f"{column}_label"] = labeling.label(
                ids=data[f"{column}_id"],
                unknown_label=("[unknown_" + column + "]").upper(),
            )

        # Re-order columns
        columns = list(TRIPLES_DF_COLUMNS) + old_col[3:]
        return data.loc[:, columns]

    def new_with_restriction(
        self,
        entities: Union[None, Collection[int], Collection[str]] = None,
        relations: Union[None, Collection[int], Collection[str]] = None,
        invert_entity_selection: bool = False,
        invert_relation_selection: bool = False,
    ) -> "TriplesFactory":  # noqa: D102
        if entities is None and relations is None:
            return self
        if entities is not None:
            entities = self.entities_to_ids(entities=entities)
        if relations is not None:
            relations = self.relations_to_ids(relations=relations)
        return (
            super()
            .new_with_restriction(
                entities=entities,
                relations=relations,
                invert_entity_selection=invert_entity_selection,
                invert_relation_selection=invert_relation_selection,
            )
            .with_labels(entity_to_id=self.entity_to_id, relation_to_id=self.relation_to_id)
        )

    def map_triples(self, triples: LabeledTriples) -> MappedTriples:
        """Convert label-based triples to ID-based triples."""
        return _map_triples_elements_to_ids(
            triples=triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
        )


def cat_triples(*triples_factories: CoreTriplesFactory) -> MappedTriples:
    """Concatenate several triples factories."""
    return torch.cat([factory.mapped_triples for factory in triples_factories], dim=0)


def splits_steps(a: Sequence[CoreTriplesFactory], b: Sequence[CoreTriplesFactory]) -> int:
    """Compute the number of moves to go from the first sequence of triples factories to the second.

    :return: The number of triples present in the training sets in both
    """
    if len(a) != len(b):
        raise ValueError("Must have same number of triples factories")

    train_1 = triple_tensor_to_set(a[0].mapped_triples)
    train_2 = triple_tensor_to_set(b[0].mapped_triples)

    # FIXME currently the implementation does not consider the non-training (i.e., second-last entries)
    #  for the number of steps. Consider more interesting way to discuss splits w/ valid

    return len(train_1.symmetric_difference(train_2))


def splits_similarity(a: Sequence[CoreTriplesFactory], b: Sequence[CoreTriplesFactory]) -> float:
    """Compute the similarity between two datasets' splits.

    :return: The number of triples present in the training sets in both
    """
    steps = splits_steps(a, b)
    n = sum(tf.num_triples for tf in a)
    return 1 - steps / n


def normalize_path(path: Union[str, pathlib.Path, TextIO]) -> pathlib.Path:
    """Normalize path."""
    if isinstance(path, TextIO):
        return pathlib.Path(path.name).resolve()
    elif isinstance(path, str):
        return pathlib.Path(path).resolve()
    elif isinstance(path, pathlib.Path):
        return path.resolve()
    else:
        raise TypeError(f"path is invalid type: {type(path)}")
