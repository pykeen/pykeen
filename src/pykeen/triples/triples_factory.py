"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

import dataclasses
import logging
import pathlib
import warnings
from collections.abc import Callable, Collection, Iterable, Mapping, MutableMapping, Sequence
from typing import (
    Any,
    ClassVar,
    TextIO,
    Union,
    cast,
)
import re
from collections.abc import Callable, Collection, Iterable, Mapping, MutableMapping, Sequence
from typing import Any, ClassVar, TextIO, cast

import numpy as np
import pandas as pd
import torch
from typing_extensions import Self

from .splitting import split, split_fully_inductive, split_semi_inductive
from .utils import TRIPLES_DF_COLUMNS, load_triples, tensor_to_df
from ..constants import COLUMN_LABELS
from ..typing import (
    BoolTensor,
    EntityMapping,
    LabeledTriples,
    LongTensor,
    MappedTriples,
    RelationMapping,
    TorchRandomHint,
)
from ..utils import (
    ExtraReprMixin,
    compact_mapping,
    format_relative_comparison,
    get_edge_index,
    invert_mapping,
    normalize_path,
    triple_tensor_to_set,
)

__all__ = [
    "KGInfo",
    "CoreTriplesFactory",
    "TriplesFactory",
    "create_entity_mapping",
    "create_relation_mapping",
    "cat_triples",
    "splits_steps",
    "splits_similarity",
    "AnyTriples",
    "get_mapped_triples",
]

logger = logging.getLogger(__name__)


def create_entity_mapping(triples: LabeledTriples) -> EntityMapping:
    """Create mapping from entity labels to IDs.

    :param triples: shape: (n, 3), dtype: str
    :returns:
        A mapping of entity labels to indices
    """
    # Split triples
    heads, tails = triples[:, 0], triples[:, 2]
    # Sorting ensures consistent results when the triples are permuted
    entity_labels = sorted(set(heads).union(tails))
    # Create mapping
    return {str(label): i for (i, label) in enumerate(entity_labels)}


def create_relation_mapping(relations: Iterable[str]) -> RelationMapping:
    """Create mapping from relation labels to IDs.

    :param relations: A set of relation labels
    :returns:
        A mapping of relation labels to indices
    """
    # Sorting ensures consistent results when the triples are permuted
    relation_labels = sorted(relations)
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
    columns: int | Collection[int],
    invert: bool = False,
    max_id: int | None = None,
) -> BoolTensor:
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
    labels_or_ids: Collection[int] | Collection[str],
    label_to_id: Mapping[str, int],
) -> Collection[int]:
    """Convert labels to IDs."""
    return [label_to_id[l_or_i] if isinstance(l_or_i, str) else l_or_i for l_or_i in labels_or_ids]


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
        ids: int | Sequence[int] | np.ndarray | LongTensor,
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

    @property
    def max_id(self) -> int:
        """Return the maximum ID (excl.)."""
        return max(self.label_to_id.values()) + 1

    def all_labels(self) -> np.ndarray:
        """Get all labels, in order."""
        return self.label(range(self.max_id))


def restrict_triples(
    mapped_triples: MappedTriples,
    entities: Collection[int] | None = None,
    relations: Collection[int] | None = None,
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


class KGInfo(ExtraReprMixin):
    """An object storing information about the number of entities and relations."""

    #: the number of unique entities
    num_entities: int

    #: the number of relations
    num_relations: int

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
    ) -> None:
        """
        Initialize the information object.

        :param num_entities:
            the number of entities.
        :param num_relations:
            the number of relations.
        """
        self.num_entities = num_entities
        self.num_relations = num_relations

    def iter_extra_repr(self) -> Iterable[str]:
        """Iterate over extra_repr components."""
        yield from super().iter_extra_repr()
        yield f"num_entities={self.num_entities}"
        yield f"num_relations={self.num_relations}"


def max_value(x: LongTensor) -> int | None:
    """Return the maximum value, or None if the tensor is empty."""
    if x.numel():
        return x.max().item()
    return None


def get_num_ids(x: LongTensor) -> int:
    """Return the number of ids values."""
    max_id = max_value(x)
    if max_id is None:
        return 0
    return max_id + 1


def _make_condensation_map(x: LongTensor) -> LongTensor | None:
    """Create a dense vector suitable for condensing Ids to a consecutive ID range."""
    # TODO: we have this functionality somewhere already?!
    if not x.numel():
        return None
    k = get_num_ids(x)
    unique_entities = x.unique()
    old_ids_t = torch.arange(k)
    if torch.equal(unique_entities, old_ids_t):
        return None
    y = torch.full((k,), fill_value=-1)
    return y.scatter_(dim=0, index=unique_entities, src=old_ids_t)


def _iter_index_remap_from_condensation_map(c: LongTensor) -> Iterable[tuple[int, int]]:
    """Iterate over pairs of old-index -> new-index."""
    old_indices = (c >= 0).nonzero().view(-1).tolist()
    new_indices = range(get_num_ids(c))
    return zip(old_indices, new_indices, strict=True)


def _maybe_condense(x: LongTensor, condensation: LongTensor | None, num: int) -> tuple[int, LongTensor]:
    """Apply condensation, if present."""
    if condensation is None:
        return num, x
    x = condensation[x]
    num = get_num_ids(x)
    return num, x


class CoreTriplesFactory(KGInfo):
    """Create instances from ID-based triples."""

    triples_file_name: ClassVar[str] = "numeric_triples.tsv.gz"
    base_file_name: ClassVar[str] = "base.pth"

    def __init__(
        self,
        mapped_triples: MappedTriples | np.ndarray,
        num_entities: int,
        num_relations: int,
        metadata: Mapping[str, Any] | None = None,
    ):
        """
        Create the triples factory.

        :param mapped_triples: shape: (n, 3)
            A three-column matrix where each row are the head identifier, relation identifier, then tail identifier.
        :param num_entities:
            The number of entities.
        :param num_relations:
            The number of relations.
        :param metadata:
            Arbitrary metadata to go with the graph

        :raises TypeError:
            if the mapped_triples are of non-integer dtype
        :raises ValueError:
            if the mapped_triples are of invalid shape
        """
        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
        )
        # ensure torch.Tensor
        mapped_triples = torch.as_tensor(mapped_triples)
        # input validation
        if mapped_triples.ndim != 2 or mapped_triples.shape[1] != 3:
            raise ValueError(f"Invalid shape for mapped_triples: {mapped_triples.shape}; must be (n, 3)")
        if mapped_triples.is_complex() or mapped_triples.is_floating_point():
            raise TypeError(f"Invalid type: {mapped_triples.dtype}. Must be integer dtype.")
        # always store as torch.long, i.e., torch's default integer dtype
        self.mapped_triples = mapped_triples.to(dtype=torch.long)
        if metadata is None:
            metadata = dict()
        self.metadata = metadata

    @classmethod
    def create(
        cls,
        mapped_triples: MappedTriples,
        num_entities: int | None = None,
        num_relations: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        """
        Create a triples factory without any label information.

        :param mapped_triples: shape: (n, 3)
            The ID-based triples.
        :param num_entities:
            The number of entities. If not given, inferred from mapped_triples.
        :param num_relations:
            The number of relations. If not given, inferred from mapped_triples.
        :param metadata:
            Additional metadata to store in the factory.

        :return:
            A new triples factory.
        """
        if num_entities is None:
            num_entities = get_num_ids(mapped_triples[:, [0, 2]])
        if num_relations is None:
            num_relations = get_num_ids(mapped_triples[:, 1])
        return cls(
            mapped_triples=mapped_triples,
            num_entities=num_entities,
            num_relations=num_relations,
            metadata=metadata,
        )

    def __eq__(self, __o: object) -> bool:  # noqa: D105
        if not isinstance(__o, CoreTriplesFactory):
            return False
        return (
            (self.num_entities == __o.num_entities)
            and (self.num_relations == __o.num_relations)
            and (self.num_triples == __o.num_triples)
            and bool((self.mapped_triples == __o.mapped_triples).all().item())
        )

    @property
    def num_triples(self) -> int:  # noqa: D401
        """The number of triples."""
        return self.mapped_triples.shape[0]

    def iter_extra_repr(self) -> Iterable[str]:
        """Iterate over extra_repr components."""
        yield from super().iter_extra_repr()
        yield f"num_triples={self.num_triples}"
        for k, v in sorted(self.metadata.items()):
            if isinstance(v, str | pathlib.Path):
                v = f'"{v}"'
            yield f"{k}={v}"

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
            metadata=self.metadata,
        )

    def get_most_frequent_relations(self, n: int | float) -> set[int]:
        """Get the IDs of the n most frequent relations.

        :param n:
            Either the (integer) number of top relations to keep or the (float) percentage of top relationships to keep.
        :returns:
            A set of IDs for the n most frequent relations
        :raises TypeError:
            If the n is the wrong type
        """
        logger.info(f"applying cutoff of {n} to {self}")
        if isinstance(n, float):
            assert 0 < n < 1
            n = int(self.num_relations * n)
        elif not isinstance(n, int):
            raise TypeError("n must be either an integer or a float")

        uniq, counts = self.mapped_triples[:, 1].unique(return_counts=True)
        top_ids = counts.topk(k=n, largest=True)[1]
        return set(uniq[top_ids].tolist())

    def clone_and_exchange_triples(
        self,
        mapped_triples: MappedTriples,
        extra_metadata: dict[str, Any] | None = None,
        keep_metadata: bool = True,
    ) -> Self:
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

        :return:
            The new factory.
        """
        return CoreTriplesFactory(
            mapped_triples=mapped_triples,
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            metadata={
                **(extra_metadata or {}),
                **(self.metadata if keep_metadata else {}),  # type: ignore
            },
        )

    def condense(self, entities: bool = True, relations: bool = False) -> Self:
        """
        Drop all IDs which are not present in the triples.

        :param entities:
            Whether to condense entity IDs.
        :param relations:
            Whether to condense relation IDs.

        .. warning::
            This creates a triples factory that may have a new entity or relation to id mapping.

        :return:
            A condensed version with potentially smaller num_entities or num_relations.
        """
        ht = self.mapped_triples[:, 0::2]
        r = self.mapped_triples[:, 1]
        # determine condensation maps (dense vectors for vectorized remapping)
        entity_condensation = _make_condensation_map(ht) if entities else None
        relation_condensation = _make_condensation_map(r) if relations else None
        # short-circuit if nothing needs to change
        if entity_condensation is None and relation_condensation is None:
            return self
        # maybe condense entities
        num_entities, ht = _maybe_condense(ht, condensation=entity_condensation, num=self.num_entities)
        # maybe condense relations
        num_relations, r = _maybe_condense(r, condensation=relation_condensation, num=self.num_relations)
        # build new triples factory
        return self.__class__(
            mapped_triples=torch.stack([ht[:, 0], r, ht[0:, 1]], dim=-1),
            num_entities=num_entities,
            num_relations=num_relations,
            create_inverse_triples=self.create_inverse_triples,
            metadata=self.metadata,
        )

    def split(
        self,
        ratios: float | Sequence[float] = 0.8,
        *,
        random_state: TorchRandomHint = None,
        randomize_cleanup: bool = False,
        method: str | None = None,
    ) -> list[Self]:
        """Split a triples factory into a training part and a variable number of (transductive) evaluation parts.

        .. warning::

            This method is not suitable to create *inductive* splits.

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
            This parameter is forwarded to the underlying :func:`pykeen.triples.splitting.split`.
        :param method:
            This parameter is forwarded to the underlying :func:`pykeen.triples.splitting.split`.


        :return:
            A partition of triples, which are split (approximately) according to the ratios, stored TriplesFactory's
            which share everything else with this root triples factory.

        .. seealso::
            :func:`pykeen.triples.splitting.split`

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
            )
            for triples in split(
                mapped_triples=self.mapped_triples,
                ratios=ratios,
                random_state=random_state,
                randomize_cleanup=randomize_cleanup,
                method=method,
            )
        ]

    def split_semi_inductive(
        self,
        ratios: float | Sequence[float] = 0.8,
        *,
        random_state: TorchRandomHint = None,
    ) -> list[Self]:
        """Create a semi-inductive split.

        In a semi-inductive split, we first split the entities into training and evaluation entities.
        The training graph is then composed of all triples involving only training entities.
        The evaluation graphs are built by looking at the triples that involve exactly one training
        and one evaluation entity.

        :param ratios:
            The *entity* split ratio(s).
        :param random_state:
            The random state used to shuffle and split the triples.

        :return:
            A partition of triples, which are split (approximately) according to the ratios, stored TriplesFactory's
            which share everything else with this root triples factory.

        .. seealso::
            - [ali2021]_
        """
        # Make new triples factories for each group
        return [
            self.clone_and_exchange_triples(
                mapped_triples=triples,
                # do not explicitly create inverse triples for testing; this is handled by the evaluation code
                create_inverse_triples=None if i == 0 else False,
            )
            for i, triples in enumerate(
                split_semi_inductive(mapped_triples=self.mapped_triples, ratios=ratios, random_state=random_state)
            )
        ]

    def split_fully_inductive(
        self,
        entity_split_train_ratio: float = 0.5,
        evaluation_triples_ratios: float | Sequence[float] = 0.8,
        random_state: TorchRandomHint = None,
    ) -> list[Self]:
        """Create a fully inductive split.

        In a fully inductive split, we first split the entities into two disjoint sets:
        training entities and inference entities. We use the induced subgraph of the training entities for training.
        The triples of the inference graph are then further split into inference triples and evaluation triples.

        :param entity_split_train_ratio:
            The ratio of entities to use for the training part. The remainder will be used for the
            inference/evaluation graph.
        :param evaluation_triples_ratios:
            The split ratio for the inference graph split.
        :param random_state:
            The random state used to shuffle and split the triples.

        :return:
            A (transductive) training triples factory, the inductive inference triples factory,
            as well as the evaluation triples factories.
        """
        training, inference, *evaluation = split_fully_inductive(
            mapped_triples=self.mapped_triples,
            entity_split_train_ratio=entity_split_train_ratio,
            evaluation_triples_ratios=evaluation_triples_ratios,
            random_state=random_state,
        )
        # separately condense the entity-to-id mappings for each of the graphs (training vs. inference)
        # we do *not* condense relations, because we only work in entity-inductive settings (for now).
        training_tf = self.clone_and_exchange_triples(mapped_triples=training).condense(entities=True, relations=False)
        inference_tf = self.clone_and_exchange_triples(mapped_triples=inference).condense(
            entities=True, relations=False
        )
        # do not explicitly create inverse triples for testing; this is handled by the evaluation code
        evaluation_tfs = [
            inference_tf.clone_and_exchange_triples(mapped_triples=mapped_triples, create_inverse_triples=False)
            for mapped_triples in evaluation
        ]
        # Make new triples factories for each group
        return [training_tf, inference_tf] + evaluation_tfs

    def merge(self, *others: Self) -> Self:
        """Merge the triples factory with others.

        The other triples factories have to be compatible.

        :param others:
            The other factories.

        :return:
            A new factory with the combined triples.

        :raises ValueError:
            If any of the other factories has incompatible settings
            (number of entities or relations, or creation of inverse triples.)
        """
        if not others:
            return self
        mapped_triples = [self.mapped_triples]
        for i, other in enumerate(others):
            if other.num_entities != self.num_entities:
                raise ValueError(
                    f"Number of entities does not match for others[{i}]: {self.num_entities=} vs. {other.num_entities=}"
                )
            if other.num_relations != self.num_relations:
                raise ValueError(
                    f"Number of relations does not match for others[{i}]: "
                    f"{self.num_relations=} vs. {other.num_relations=}"
                )
            if other.create_inverse_triples != self.create_inverse_triples:
                raise ValueError(
                    f"Creation of inverse triples does not match for others[{i}]: "
                    f"{self.create_inverse_triples=} vs. {other.create_inverse_triples=}"
                )
            mapped_triples.append(other.mapped_triples)
        return self.clone_and_exchange_triples(torch.cat(mapped_triples, dim=0))

    def entities_to_ids(self, entities: Collection[int] | Collection[str]) -> Collection[int]:
        """Normalize entities to IDs.

        :param entities: A collection of either integer identifiers for entities or
            string labels for entities (that will get auto-converted)
        :returns: Integer identifiers for entities
        :raises ValueError: If the ``entities`` passed are string labels
            and this triples factory does not have an entity label to identifier mapping
            (e.g., it's just a base :class:`CoreTriplesFactory` instance)
        """
        for e in entities:
            if not isinstance(e, int):
                raise ValueError(f"{self.__class__.__name__} cannot convert entity IDs from {type(e)} to int.")
        return cast(Collection[int], entities)

    def relations_to_ids(self, relations: Collection[int] | Collection[str]) -> Collection[int]:
        """Normalize relations to IDs.

        :param relations: A collection of either integer identifiers for relations or
            string labels for relations (that will get auto-converted)
        :returns: Integer identifiers for relations
        :raises ValueError: If the ``relations`` passed are string labels
            and this triples factory does not have a relation label to identifier mapping
            (e.g., it's just a base :class:`CoreTriplesFactory` instance)
        """
        for e in relations:
            if not isinstance(e, int):
                raise ValueError(f"{self.__class__.__name__} cannot convert relation IDs from {type(e)} to int.")
        return cast(Collection[int], relations)

    def get_mask_for_relations(
        self,
        relations: Collection[int],
        invert: bool = False,
    ) -> BoolTensor:
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
        tensor: LongTensor,
        **kwargs: torch.Tensor | np.ndarray | Sequence,
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
        entities: None | Collection[int] | Collection[str] = None,
        relations: None | Collection[int] | Collection[str] = None,
        invert_entity_selection: bool = False,
        invert_relation_selection: bool = False,
    ) -> Self:
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
    # docstr-coverage: inherited
    def from_path_binary(
        cls,
        path: str | pathlib.Path | TextIO,
    ) -> Self:  # noqa: D102
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
        # TODO: consider restricting metadata to JSON
        data = dict(torch.load(path.joinpath(cls.base_file_name), weights_only=False))
        # load numeric triples
        data["mapped_triples"] = torch.as_tensor(
            pd.read_csv(path.joinpath(cls.triples_file_name), sep="\t", dtype=int).values,
            dtype=torch.long,
        )
        return data

    def to_path_binary(
        self,
        path: str | pathlib.Path | TextIO,
    ) -> pathlib.Path:
        """
        Save triples factory to path in (PyTorch's .pt) binary format.

        :param path:
            The path to store the triples factory to.
        :returns:
            The path to the file that got dumped
        """
        path = normalize_path(path, mkdir=True)

        # store numeric triples
        pd.DataFrame(
            data=self.mapped_triples.numpy(),
            columns=COLUMN_LABELS,
        ).to_csv(path.joinpath(self.triples_file_name), sep="\t", index=False)

        # store metadata
        torch.save(self._get_binary_state(), path.joinpath(self.base_file_name))
        logger.info(f"Stored {self} to {path.as_uri()}")

        return path

    def _get_binary_state(self):
        return dict(
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            metadata=self.metadata,
        )


def _maybe_condense_map(id_to_label: Mapping[int, str], condensation: LongTensor | None) -> Mapping[str, int]:
    """Condense label to Id mapping, if necessary."""
    if condensation is None:
        return {label: idx for idx, label in id_to_label.items()}
    return {id_to_label[old]: new for old, new in _iter_index_remap_from_condensation_map(condensation)}


class TriplesFactory(CoreTriplesFactory):
    """Create instances given the path to triples."""

    file_name_entity_to_id: ClassVar[str] = "entity_to_id"
    file_name_relation_to_id: ClassVar[str] = "relation_to_id"

    def __init__(
        self,
        mapped_triples: MappedTriples | np.ndarray,
        entity_to_id: EntityMapping,
        relation_to_id: RelationMapping,
        metadata: Mapping[str, Any] | None = None,
        num_entities: int | None = None,
        num_relations: int | None = None,
    ):
        """
        Create the triples factory.

        :param mapped_triples: shape: (n, 3)
            A three-column matrix where each row are the head identifier, relation identifier, then tail identifier.
        :param entity_to_id:
            The mapping from entities' labels to their indices.
        :param relation_to_id:
            The mapping from relations' labels to their indices.
        :param metadata:
            Arbitrary metadata to go with the graph
        :param num_entities:
            the number of entities. May be None, in which case this number is inferred by the label mapping
        :param num_relations:
            the number of relations. May be None, in which case this number is inferred by the label mapping

        :raises ValueError:
            if the explicitly provided number of entities or relations does not match with the one given
            by the label mapping
        """
        self.entity_labeling = Labeling(label_to_id=entity_to_id)
        if num_entities is None:
            num_entities = self.entity_labeling.max_id
        elif num_entities != self.entity_labeling.max_id:
            raise ValueError(
                f"Mismatch between the number of entities in labeling ({self.entity_labeling.max_id}) "
                f"vs. explicitly provided num_entities={num_entities}",
            )
        self.relation_labeling = Labeling(label_to_id=relation_to_id)
        if num_relations is None:
            num_relations = self.relation_labeling.max_id
        elif num_relations != self.relation_labeling.max_id:
            raise ValueError(
                f"Mismatch between the number of relations in labeling ({self.relation_labeling.max_id}) "
                f"vs. explicitly provided num_relations={num_relations}",
            )
        super().__init__(
            mapped_triples=mapped_triples,
            num_entities=self.entity_labeling.max_id,
            num_relations=self.relation_labeling.max_id,
            metadata=metadata,
        )

    @classmethod
    def from_labeled_triples(
        cls,
        triples: LabeledTriples,
        *,
        entity_to_id: EntityMapping | None = None,
        relation_to_id: RelationMapping | None = None,
        compact_id: bool = True,
        filter_out_candidate_inverse_relations: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        """
        Create a new triples factory from label-based triples.

        :param triples: shape: (n, 3), dtype: str
            The label-based triples.
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
            metadata=metadata,
        )

    @classmethod
    def from_path(
        cls,
        path: str | pathlib.Path | TextIO,
        *,
        entity_to_id: EntityMapping | None = None,
        relation_to_id: RelationMapping | None = None,
        compact_id: bool = True,
        metadata: dict[str, Any] | None = None,
        load_triples_kwargs: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> Self:
        """
        Create a new triples factory from triples stored in a file.

        :param path:
            The path where the label-based triples are stored.
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
        :param kwargs:
            additional keyword-based parameters, which are ignored.

        :return:
            A new triples factory.
        """
        path = normalize_path(path)

        # TODO: Check if lazy evaluation would make sense
        triples = load_triples(path, **(load_triples_kwargs or {}))

        return cls.from_labeled_triples(
            triples=triples,
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

    # docstr-coverage: inherited
    def condense(self, entities: bool = True, relations: bool = False) -> Self:  # noqa: D102
        ht = self.mapped_triples[:, 0::2]
        r = self.mapped_triples[:, 1]
        # determine condensation maps (dense vectors for vectorized remapping)
        entity_condensation = _make_condensation_map(ht) if entities else None
        relation_condensation = _make_condensation_map(r) if relations else None
        # short-circuit if nothing needs to change
        if entity_condensation is None and relation_condensation is None:
            return self
        # maybe condense entities
        num_entities, ht = _maybe_condense(ht, condensation=entity_condensation, num=self.num_entities)
        entity_to_id = _maybe_condense_map(self.entity_id_to_label, condensation=entity_condensation)
        # maybe condense relations
        num_relations, r = _maybe_condense(r, condensation=relation_condensation, num=self.num_relations)
        relation_to_id = _maybe_condense_map(self.relation_id_to_label, condensation=relation_condensation)
        return self.__class__(
            mapped_triples=torch.stack([ht[:, 0], r, ht[0:, 1]], dim=-1),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            num_entities=num_entities,
            num_relations=num_relations,
            create_inverse_triples=self.create_inverse_triples,
            metadata=self.metadata,
        )

    # docstr-coverage: inherited
    def merge(self, *others: Self) -> Self:  # noqa: D102
        for i, other in enumerate(others):
            if other.entity_to_id != self.entity_to_id:
                raise ValueError(
                    f"Entity to ID mapping does not match for others[{i}]: "
                    f"{self.entity_to_id=} vs. {other.entity_to_id=}"
                )
            if other.relation_to_id != self.relation_to_id:
                raise ValueError(
                    f"Relation to ID mapping does not match for others[{i}]: "
                    f"{self.relation_to_id=} vs. {other.relation_to_id=}"
                )
        return super().merge(*others)

    def to_core_triples_factory(self) -> CoreTriplesFactory:
        """Return this factory as a core factory."""
        return CoreTriplesFactory(
            mapped_triples=self.mapped_triples,
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            metadata=self.metadata,
        )

    # docstr-coverage: inherited
    def to_path_binary(self, path: str | pathlib.Path | TextIO) -> pathlib.Path:  # noqa: D102
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
            pd.DataFrame(
                data=data.items(),
                columns=["label", "id"],
            ).sort_values(by="id").set_index("id").to_csv(
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
            data[name] = dict(zip(df["label"], df["id"], strict=False))
        return data

    # docstr-coverage: inherited
    def _get_binary_state(self):  # noqa: D102
        return dict(
            mapped_triples=self.mapped_triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            metadata=self.metadata,
        )

    # docstr-coverage: inherited
    def clone_and_exchange_triples(
        self,
        mapped_triples: MappedTriples,
        extra_metadata: dict[str, Any] | None = None,
        keep_metadata: bool = True,
    ) -> Self:  # noqa: D102
        return TriplesFactory(
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            mapped_triples=mapped_triples,
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

    def label_triples(
        self,
        triples: MappedTriples,
        unknown_entity_label: str = "[UNKNOWN]",
        unknown_relation_label: str | None = None,
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
                    triples.t().numpy(), strict=False,
                )
            ],
            axis=1,
        )

    # docstr-coverage: inherited
    def entities_to_ids(self, entities: Collection[int] | Collection[str]) -> Collection[int]:  # noqa: D102
        return _ensure_ids(labels_or_ids=entities, label_to_id=self.entity_labeling.label_to_id)

    # docstr-coverage: inherited
    def relations_to_ids(self, relations: Collection[int] | Collection[str]) -> Collection[int]:  # noqa: D102
        return _ensure_ids(labels_or_ids=relations, label_to_id=self.relation_labeling.label_to_id)

    def get_mask_for_relations(
        self,
        relations: Collection[int] | Collection[str],
        invert: bool = False,
    ) -> BoolTensor:
        """Get a boolean mask for triples with the given relations."""
        return super().get_mask_for_relations(relations=self.relations_to_ids(relations=relations))

    def entity_word_cloud(self, top: int | None = None):
        """Make a word cloud based on the frequency of occurrence of each entity in a Jupyter notebook.

        :param top: The number of top entities to show. Defaults to 100.
        :returns: A word cloud object for a Jupyter notebook

        .. warning::

            This function requires the ``wordcloud`` package. Use ``pip install pykeen[wordcloud]`` to install it.
        """
        return self._word_cloud(
            ids=get_edge_index(mapped_triples=self.mapped_triples).t(),
            id_to_label=self.entity_labeling.id_to_label,
            top=top or 100,
        )

    def relation_word_cloud(self, top: int | None = None):
        """Make a word cloud based on the frequency of occurrence of each relation in a Jupyter notebook.

        :param top: The number of top relations to show. Defaults to 100.
        :returns: A world cloud object for a Jupyter notebook

        .. warning::

            This function requires the ``wordcloud`` package. Use ``pip install pykeen[wordcloud]`` to install it.
        """
        return self._word_cloud(
            ids=self.mapped_triples[:, 1],
            id_to_label=self.relation_labeling.id_to_label,
            top=top or 100,
        )

    def _word_cloud(self, *, ids: LongTensor, id_to_label: Mapping[int, str], top: int):
        try:
            from wordcloud import WordCloud
        except ImportError:
            logger.warning(
                "Could not import module `wordcloud`. Try installing it with `pip install wordcloud`",
            )
            return

        # pre-filter to keep only topk
        uniq, counts = ids.reshape(-1).unique(return_counts=True)

        # if top is larger than the number of available options
        top = min(top, uniq.numel())
        top_counts, top_ids = counts.topk(k=top, largest=True)

        # Generate a word cloud image
        svg_str: str = (
            WordCloud(normalize_plurals=False, max_words=top, mode="RGBA", background_color=None)
            .generate_from_frequencies(
                frequencies=dict(zip(map(id_to_label.__getitem__, top_ids.tolist()), top_counts.tolist(), strict=False))
            )
            .to_svg()
        )

        from IPython.core.display import SVG

        return SVG(data=svg_str)

    # docstr-coverage: inherited
    def tensor_to_df(
        self,
        tensor: LongTensor,
        **kwargs: torch.Tensor | np.ndarray | Sequence,
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

    # docstr-coverage: inherited
    def new_with_restriction(
        self,
        entities: None | Collection[int] | Collection[str] = None,
        relations: None | Collection[int] | Collection[str] = None,
        invert_entity_selection: bool = False,
        invert_relation_selection: bool = False,
    ) -> Self:  # noqa: D102
        if entities is None and relations is None:
            return self
        if entities is not None:
            entities = self.entities_to_ids(entities=entities)
        if relations is not None:
            relations = self.relations_to_ids(relations=relations)
        tf = super().new_with_restriction(
            entities=entities,
            relations=relations,
            invert_entity_selection=invert_entity_selection,
            invert_relation_selection=invert_relation_selection,
        )
        tf.entity_labeling = self.entity_labeling
        tf.relation_labeling = self.relation_labeling
        return tf

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

    :param a: A sequence of triples factories
    :param b: A sequence of triples factories
    :return: The number of triples present in the training sets in both
    :raises ValueError: If the sequences of triples factories are a different length
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

    :param a: A sequence of triples factories
    :param b: A sequence of triples factories
    :return: The number of triples present in the training sets in both
    """
    steps = splits_steps(a, b)
    n = sum(tf.num_triples for tf in a)
    return 1 - steps / n


AnyTriples = tuple[str, str, str] | Sequence[tuple[str, str, str]] | LabeledTriples | MappedTriples | CoreTriplesFactory


def get_mapped_triples(
    x: AnyTriples | None = None,
    *,
    mapped_triples: MappedTriples | None = None,
    triples: None | LabeledTriples | tuple[str, str, str] | Sequence[tuple[str, str, str]] = None,
    factory: CoreTriplesFactory | None = None,
) -> MappedTriples:
    """
    Get ID-based triples either directly, or from a factory.

    Preference order:
    1. `mapped_triples`
    2. `triples` (converted using factory)
    3. `x`
    4. `factory.mapped_triples`

    :param x:
        either of label-based triples, ID-based triples, a factory, or None.
    :param mapped_triples: shape: (n, 3)
        the ID-based triples
    :param triples:
        the label-based triples
    :param factory:
        the triples factory

    :raises ValueError:
        if all inputs are None, or provided inputs are invalid.

    :return:
        the ID-based triples
    """
    # ID-based triples
    if mapped_triples is not None:
        if torch.is_floating_point(mapped_triples):
            raise ValueError(
                f"mapped_triples must be on long (or compatible) data type, but are {mapped_triples.dtype}"
            )
        if mapped_triples.ndim != 2 or mapped_triples.shape[1] != 3:
            raise ValueError(f"mapped_triples must be of shape (?, 3), but are {mapped_triples.shape}")
        return mapped_triples

    # labeled triples
    if triples is not None:
        if factory is None or not isinstance(factory, TriplesFactory):
            raise ValueError("If triples are not ID-based, a triples factory must be provided and label-based.")

        # make sure triples are a numpy array
        triples = np.asanyarray(triples)

        # make sure triples are 2d
        triples = np.atleast_2d(triples)

        # convert to ID-based
        return factory.map_triples(triples)

    # triples factory
    if x is None and factory is not None:
        return factory.mapped_triples

    # all keyword-based options have been none
    if x is None:
        raise ValueError("All parameters were None.")

    if isinstance(x, torch.Tensor):
        # delegate to keyword-based get_mapped_triples to re-use optional validation logic
        return get_mapped_triples(mapped_triples=x)

    if isinstance(x, CoreTriplesFactory):
        # delegate to keyword-based get_mapped_triples to re-use optional validation logic
        return get_mapped_triples(mapped_triples=x.mapped_triples)

    # only labeled triples are remaining
    return get_mapped_triples(triples=x, factory=factory)
