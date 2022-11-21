# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates instances of hyper-relational statements."""
import functools
import logging
import pathlib
import re
from abc import abstractmethod
from collections import defaultdict
from typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import torch
from tqdm import tqdm

from .instances import LCWAInstances, SLCWAInstances
from .triples_factory import KGInfo
from .utils import STATEMENT_PADDING, load_rdf_star_statements
from ..typing import EntityMapping, LabeledTriples, MappedTriples, RelationMapping, TorchRandomHint
from ..utils import (
    ExtraReprMixin,
    compact_mapping,
    format_relative_comparison,
    get_edge_index,
    invert_mapping,
    normalize_path,
    triple_tensor_to_set,
)

logger = logging.getLogger(__name__)

INVERSE_SUFFIX = "_inverse"
TRIPLES_DF_COLUMNS = ("head_id", "head_label", "relation_id", "relation_label", "tail_id", "tail_label")


def _map_statements(
    statements: np.ndarray,
    entity_to_id,
    relation_to_id,
) -> Tuple[torch.LongTensor, Collection[np.ndarray]]:
    if statements.size == 0:
        logger.warning("Provided empty statements to map.")
        return torch.empty(0, 3, dtype=torch.long)

    # When entities/relations that don't exist are trying to be mapped, they get the id "-1"
    entity_getter = np.vectorize(entity_to_id.get)
    entity_column = entity_getter(statements[:, 0::2], [-1])

    relation_getter = np.vectorize(relation_to_id.get)
    relation_column = relation_getter(statements[:, 1::2], [-1])

    # Filter statements with non-mapped entities/relations
    entity_filter = entity_column < 0
    relation_filter = relation_column < 0
    num_no_entity = entity_filter.any(axis=-1).sum()
    num_no_relation = relation_filter.any(axis=-1).sum()

    if num_no_entity > 0 or num_no_relation > 0:
        logger.warning(
            f"You're trying to map statements with {num_no_entity} entities and {num_no_relation} relations"
            f" that are not in the training set. These statements will be excluded from the mapping.",
        )
        # if entity/relation is not mapped (in the main triple OR qualifiers) - we remove that statement entirely
        # TODO if an unmapped e/r is in a qualifier, we might remove only this particular pair (cumbersome)
        non_mappable_statements = entity_filter.any(axis=-1) | relation_filter.any(axis=-1)
        entity_column = entity_column[~non_mappable_statements]
        relation_column = relation_column[~non_mappable_statements]
        logger.warning(
            f"In total {non_mappable_statements.sum():.0f} from {statements.shape[0]:.0f} triples were filtered out",
        )

    mapped_statements = np.empty(
        (entity_column.shape[0], entity_column.shape[1] + relation_column.shape[1]), dtype=np.int64
    )
    mapped_statements[:, 0::2] = entity_column
    mapped_statements[:, 1::2] = relation_column

    # Note: Unique changes the order of the triples
    # Note: Using unique means implicit balancing of training samples
    unique_mapped_statements = np.unique(ar=mapped_statements, axis=0)
    return torch.tensor(unique_mapped_statements, dtype=torch.long)


class StatementFactory(KGInfo):
    """A mix of CoreTriplesFactory and TriplesFactory, WIP"""

    triples_file_name: ClassVar[str] = "numeric_statements.tsv.gz"
    base_file_name: ClassVar[str] = "base.pth"

    def __init__(
        self,
        mapped_statements: Union[MappedTriples, np.ndarray, torch.tensor],
        entity_to_id: Dict,
        relation_to_id: Dict,
        create_inverse_triples: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        max_num_qualifier_pairs: int = None,
    ) -> None:
        """
        Create the statement factory.

        :param mapped_statements: shape: (n, max_num_qualifier_pairs)
            A matrix where each row is a statement (h, r, t, {qr1, qe2}_i).
        :param num_entities:
            The number of entities.
        :param num_relations:
            The number of relations.
        :param create_inverse_triples:
            Whether to create inverse triples.
        :param metadata:
            Arbitrary metadata to go with the graph
        :param max_num_qualifier_pairs:
            Maximum number of qualifier pairs to load

        :raises TypeError:
            if the mapped_statements are of non-integer dtype
        :raises ValueError:
            if the mapped_statements are of invalid shape
        """
        num_entities = len(entity_to_id)
        num_relations = len(relation_to_id)
        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            create_inverse_triples=create_inverse_triples,
        )
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id
        # ensure torch.Tensor
        mapped_statements = torch.as_tensor(mapped_statements)
        # input validation
        if mapped_statements.ndim != 2 or mapped_statements.shape[1] != (max_num_qualifier_pairs * 2 + 3):
            raise ValueError(
                f"Invalid shape for mapped_triples: {mapped_statements.shape}; "
                f"must be (n, 3 + max_num_qualifier_pairs * 2)"
            )
        if mapped_statements.is_complex() or mapped_statements.is_floating_point():
            raise TypeError(f"Invalid type: {mapped_statements.dtype}. Must be integer dtype.")
        # always store as torch.long, i.e., torch's default integer dtype
        self.mapped_statements = mapped_statements.to(dtype=torch.long)
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.max_num_qualifier_pairs = max_num_qualifier_pairs
        self.paddix_idx = entity_to_id["__padding__"]
        self.non_qualifier_only_entities = self.mapped_statements[:, [0, 2]].unique()

    @classmethod
    def from_path(
        cls,
        path: Union[str, pathlib.Path, TextIO],
        *,
        create_inverse_triples: bool = False,
        entity_to_id: Optional[EntityMapping] = None,
        relation_to_id: Optional[RelationMapping] = None,
        compact_id: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        load_triples_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> "StatementFactory":
        """
        Create a new statement factory from statements stored in a file.

        :param path:
            The path where the label-based statements are stored.
        :param create_inverse_triples:
            Whether to create inverse statements (qualifier relations are not changes).
        :param entity_to_id:
            The mapping from entity labels to ID. If None, create a new one from the statements.
        :param relation_to_id:
            The mapping from relations labels to ID. If None, create a new one from the statements.
        :param compact_id:
            Whether to compact IDs such that the IDs are consecutive.
        :param metadata:
            Arbitrary key/value pairs to store as metadata with the statement factory. Do not
            include ``path`` as a key because it is automatically taken from the ``path``
            kwarg to this function.
        :param load_triples_kwargs: Optional keyword arguments to pass to :func:`load_triples`.
            Could include the ``delimiter`` or a ``column_remapping``.
        :param kwargs:
            additional keyword-based parameters, which are ignored.

        :return:
            A new triples factory.
        """
        path = normalize_path(path)

        statements = load_rdf_star_statements(path, **(load_triples_kwargs or {}))

        return cls.from_labeled_statements(
            statements=statements,
            create_inverse_triples=create_inverse_triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            compact_id=compact_id,
            metadata={
                "path": path,
                **(metadata or {}),
            },
        )

    @classmethod
    def from_labeled_statements(
        cls,
        statements: LabeledTriples,
        *,
        create_inverse_triples: bool = False,
        entity_to_id: Optional[EntityMapping] = None,
        relation_to_id: Optional[RelationMapping] = None,
        compact_id: bool = True,
        filter_out_candidate_inverse_relations: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "StatementFactory":
        """
        Create a new state e t factory from label-based statements.

        :param statements: shape: (n, 3 + 2 * max_num_quals), dtype: str
            The label-based statements.
        :param create_inverse_triples:
            Whether to create inverse statements.
        :param entity_to_id:
            The mapping from entity labels to ID. If None, create a new one from the statements.
        :param relation_to_id:
            The mapping from relations labels to ID. If None, create a new one from the statements.
        :param compact_id:
            Whether to compact IDs such that the IDs are consecutive.
        :param filter_out_candidate_inverse_relations:
            Whether to remove triples with relations with the inverse suffix.
        :param metadata:
            Arbitrary key/value pairs to store as metadata

        :return:
            A new triples factory.
        """
        # Check if the statements are inverted already
        # We re-create them pure index based to ensure that _all_ inverse triples are present and that they are
        # contained if and only if create_inverse_triples is True.
        if filter_out_candidate_inverse_relations:
            unique_relations, inverse = np.unique(statements[:, 1], return_inverse=True)
            suspected_to_be_inverse_relations = {r for r in unique_relations if r.endswith(INVERSE_SUFFIX)}
            if len(suspected_to_be_inverse_relations) > 0:
                logger.warning(
                    f"Some statements already have the inverse relation suffix {INVERSE_SUFFIX}. "
                    f"Re-creating inverse statements to ensure consistency. You may disable this behaviour by passing "
                    f"filter_out_candidate_inverse_relations=False",
                )
                relation_ids_to_remove = [
                    i for i, r in enumerate(unique_relations.tolist()) if r in suspected_to_be_inverse_relations
                ]
                mask = np.isin(element=inverse, test_elements=relation_ids_to_remove, invert=True)
                logger.info(f"keeping {mask.sum() / mask.shape[0]} statements.")
                statements = statements[mask]

        # Generate entity mapping if necessary
        if entity_to_id is None:
            entity_to_id = create_statement_entity_mapping(statements=statements)
        if compact_id:
            entity_to_id = compact_mapping(mapping=entity_to_id)[0]

        # Generate relation mapping if necessary
        if relation_to_id is None:
            relation_to_id = create_statement_relation_mapping(statements[:, 1::2])
        if compact_id:
            relation_to_id = compact_mapping(mapping=relation_to_id)[0]

        # Map statements of labels to statements of IDs.
        mapped_statements = _map_statements(
            statements=statements,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

        return cls(
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            mapped_statements=mapped_statements,
            create_inverse_triples=create_inverse_triples,
            metadata=metadata,
            max_num_qualifier_pairs=(mapped_statements.shape[1] - 3) // 2,
        )

    def _process_inverse_relations(self):
        relations = list(sorted(set([e for s in self.statements for e in s[1::2]])))

        # Check if the triples are inverted already
        relations_already_inverted = self._check_already_inverted_relations(relations)

        if self.create_inverse_triples or relations_already_inverted:
            self.create_inverse_triples = True
            if relations_already_inverted:
                logger.info(
                    f"Some triples already have suffix {INVERSE_SUFFIX}. "
                    f"Creating TriplesFactory based on inverse triples",
                )
                self.relation_to_inverse = {
                    re.sub("_inverse$", "", relation): f"{re.sub('_inverse$', '', relation)}{INVERSE_SUFFIX}"
                    for relation in relations
                }

            else:
                self.relation_to_inverse = {relation: f"{relation}{INVERSE_SUFFIX}" for relation in relations}
                inverse_statements = [[s[2], self.relation_to_inverse[s[1]], s[0], *s[3:]] for s in self.statements]
                # extend original triples with inverse ones
                self.statements.extend(inverse_statements)

        else:
            self.create_inverse_triples = False
            self.relation_to_inverse = None

    @staticmethod
    def _check_already_inverted_relations(relations: Iterable[str]) -> bool:
        for relation in relations:
            if relation.endswith(INVERSE_SUFFIX):
                # We can terminate the search after finding the first inverse occurrence
                return True

        return False

    def create_data_object(self):
        """Create PyTorch Geometric data object."""

        try:
            from torch_geometric.data import Data
        except ImportError as err:
            raise ImportError("Requires `torch_geometric` to be installed.") from err

        node_features = self._create_node_feature_tensor()
        edge_index = self._create_edge_index()
        qualifier_index = self._create_qualifier_index()
        edge_type = self._create_edge_type()

        entities = torch.as_tensor(
            data=sorted(set(self.entity_to_id.values())),  # exclude padding entity
            dtype=torch.long,
        )

        return Data(
            x=node_features,
            edge_index=edge_index,
            qualifier_index=qualifier_index,
            edge_type=edge_type,
            entities=entities,
        )

    def _compose_qualifier_batch(self, keys, mapping) -> torch.LongTensor:
        # batch.shape:
        batch_size = len(keys)

        # Lookup IDs of qualifiers
        qualifier_ids = [mapping.get(tuple(k.tolist())) for k in keys]

        # Determine maximum length (for dynamic padding)
        max_len = max(map(len, qualifier_ids))

        # bound maximum length for guaranteed max memory consumption
        # TODO: num_qualifier_pairs = max_num_qualifier_pairs ?
        max_len = min(max_len, self.max_num_qualifier_pairs)

        # Allocate result
        result = torch.full(size=(batch_size, max_len), fill_value=-1, dtype=torch.long)

        # Retrieve qualifiers
        for i, this_qualifier_ids in enumerate(qualifier_ids):
            # limit number of qualifiers
            # TODO: shuffle?
            this_qualifier_ids = this_qualifier_ids[:max_len]
            result[i, : len(this_qualifier_ids)] = torch.as_tensor(data=this_qualifier_ids, dtype=torch.long)

        return result

    def _create_qualifier_index(self):
        """Create a COO matrix of shape (3, num_qualifiers). Only non-zero (non-padded) qualifiers are retained.

        row0: qualifying relations
        row1: qualifying entities
        row2: index row which connects a pair (qual_r, qual_e) to a statement index k
        :return: shape: (3, num_qualifiers)
        """
        if self.max_num_qualifier_pairs is None or self.max_num_qualifier_pairs == 0:
            return None

        qual_relations, qual_entities, qual_k = [], [], []

        # It is assumed that statements are already padded
        for triple_id, statement in enumerate(self.mapped_statements):
            qualifiers = statement[3:]
            entities = qualifiers[1::2]
            relations = qualifiers[::2]
            # Ensure that PADDING has id=0
            non_zero_rels = relations[np.nonzero(relations)]
            non_zero_ents = entities[np.nonzero(entities)]
            assert len(non_zero_rels) == len(
                non_zero_ents
            ), "Number of non-padded qualifying relations is not equal to the # of qualifying entities"
            num_qualifier_pairs = non_zero_ents.shape[0]

            for j in range(num_qualifier_pairs):
                qual_relations.append(non_zero_rels[j].item())
                qual_entities.append(non_zero_ents[j].item())
                qual_k.append(triple_id)

        qualifier_index = torch.stack(
            [
                torch.tensor(qual_relations, dtype=torch.long),
                torch.tensor(qual_entities, dtype=torch.long),
                torch.tensor(qual_k, dtype=torch.long),
            ],
            dim=0,
        )

        if self.create_inverse_triples:
            # qualifier index is the same for inverse statements
            qualifier_index[2, len(qual_relations) // 2 :] = qualifier_index[2, : len(qual_relations) // 2]

        return qualifier_index

    def _create_node_feature_tensor(self) -> torch.Tensor:
        """Create the node feature tensor."""
        if self.node_feature_tensor is not None:
            return self.node_feature_tensor
        if self.one_hot_encoding:
            return torch.eye(n=len(self.entity_to_id))
        else:
            return torch.empty(len(self.entity_to_id), self.feature_dim)

    def _create_edge_index(self) -> torch.Tensor:
        """Create edge index where first row represents the source nodes and the second row the target nodes."""
        mapped_heads = self.mapped_statements[:, 0].view(1, -1)
        mapped_tails = self.mapped_statements[:, 2].view(1, -1)
        edge_index = torch.cat([mapped_heads, mapped_tails], dim=0)
        return edge_index

    def _create_edge_type(self) -> torch.Tensor:
        """Create edge type tensor where each entry correspond to the relationship type of a triple in the dataset."""
        # Inverse triples are created in the base class

        return self.mapped_statements[:, 1]

    def create_lcwa_instances(self, use_tqdm: Optional[bool] = None) -> LCWAInstances:
        """Create LCWA instances for this factory's statements."""
        s_p_q_to_multi_tails = _create_multi_label_tails_instance(
            mapped_statements=self.mapped_statements,
            use_tqdm=use_tqdm,
        )
        spq, multi_o = zip(*s_p_q_to_multi_tails.items())
        mapped_statements: torch.LongTensor = torch.tensor(spq, dtype=torch.long)
        labels = np.array([np.array(item) for item in multi_o], dtype=object)

        # create mask
        entity_mask = torch.zeros(self.num_entities, dtype=torch.bool)
        entity_mask[self.non_qualifier_only_entities] = True

        return LCWAInstances(
            mapped_statements=mapped_statements,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            labels=labels,
            # entity_mask=entity_mask,
        )

    def create_slcwa_instances(self) -> SLCWAInstances:
        """Create sLCWA instances for this factory's statements."""
        return SLCWAInstances(
            mapped_statements=self.mapped_statements,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            entities=self.non_qualifier_only_entities,
        )

    # TODO those properties conflict with KGInfo object fields of the same name, comment for now
    # @property
    # def num_entities(self) -> int:
    #     """The number of unique entities."""
    #     return len(self.entity_to_id)
    #
    # @property
    # def num_relations(self) -> int:
    #     """The number of unique relations."""
    #     return len(self.relation_to_id)

    @property
    def statement_length(self) -> int:
        """The number of unique relations."""
        return self.max_num_qualifier_pairs * 2 + 3

    @property
    def num_statements(self) -> int:
        """The number of statements."""
        return self.mapped_statements.shape[0]

    @property
    def qualifier_ratio(self) -> float:
        """Return the percentage of statements with qualifiers."""
        return (~(self.mapped_statements[:, 3::2] == self.padding_idx).all(dim=1)).sum().item() / self.num_statements

    def extra_repr(self) -> str:
        """Extra representation."""
        return (
            f"num_entities={self.num_entities:,}, "
            f"num_relations={self.num_relations:,}, "
            f"num_statements={self.num_statements:,}, "
            f"max_num_qualifier_pairs={self.max_num_qualifier_pairs}, "
            f"qualifier_ratio={self.qualifier_ratio:.2%}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"

    @property
    def num_triples(self) -> int:
        return self.num_statements


def create_statement_entity_mapping(statements: Iterable[str]):
    entities = list(sorted(set([e for s in statements for e in s[::2]])))  # padding is already in the ndarray
    entity_to_id = {entity: id for id, entity in enumerate(entities)}
    return entity_to_id


def create_statement_relation_mapping(relations: np.ndarray):
    # Sorting ensures consistent results when the triples are permuted
    relation_labels = sorted(
        set(relations.flatten()),
        key=lambda x: (re.sub(f"{INVERSE_SUFFIX}$", "", x), x.endswith(f"{INVERSE_SUFFIX}")),
    )
    # Create mapping
    return {str(label): i for (i, label) in enumerate(relation_labels)}


def _create_multi_label_tails_instance(
    mapped_statements: torch.Tensor,
    use_tqdm: Optional[bool] = None,
) -> Dict[Tuple[int, int], List[int]]:
    """Create for each (h,r,q*) pair the multi tail label."""
    logger.debug("Creating multi label tails instance")

    """
    The mapped triples matrix has to be a numpy array to ensure correct pair hashing, as explained in
    https://github.com/pykeen/pykeen/commit/1bc71fe4eb2f24190425b0a4d0b9d6c7b9c4653a
    """
    mapped_statements = mapped_statements.cpu().detach().numpy()

    s_p_q_to_multi_tails_new = _create_multi_label_instances(
        mapped_statements,
        element_1_index=0,
        element_2_index=1,
        label_index=2,
        use_tqdm=use_tqdm,
    )

    logger.debug("Created multi label tails instance")

    return s_p_q_to_multi_tails_new


def _create_multi_label_instances(
    mapped_statements: torch.Tensor,
    element_1_index: int,
    element_2_index: int,
    label_index: int,
    use_tqdm: Optional[bool] = None,
) -> Dict[Tuple[int, ...], List[int]]:
    """Create for each (element_1, element_2) pair the multi-label."""
    instance_to_multi_label = defaultdict(set)

    if use_tqdm is None:
        use_tqdm = True

    it = mapped_statements

    if use_tqdm:
        it = tqdm(mapped_statements, unit="statement", unit_scale=True, desc="Grouping statements")

    for row in it:
        instance = tuple([row[element_1_index], row[element_2_index]] + row[3:].tolist())
        instance_to_multi_label[instance].add(row[label_index])

    # Create lists out of sets for proper numpy indexing when loading the labels
    # TODO is there a need to have a canonical sort order here?
    instance_to_multi_label_new = {key: list(value) for key, value in instance_to_multi_label.items()}

    return instance_to_multi_label_new
