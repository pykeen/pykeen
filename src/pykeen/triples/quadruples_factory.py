# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard tKG quadruples."""

import logging
import pathlib
import warnings
from datetime import datetime
from typing import (
    Any,
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
    Union,
    cast,
)

import numpy as np
import pandas as pd
import torch
from class_resolver import HintOrType, OptionalKwargs
from torch.utils.data import Dataset

from .instances import BatchedSLCWAInstances, LCWAQuadrupleInstances, SLCWAQuadrupleInstances, SubGraphSLCWAInstances
from .triples_factory import (
    INVERSE_SUFFIX,
    CoreTriplesFactory,
    Labeling,
    _ensure_ids,
    _map_triples_elements_to_ids,
    compact_mapping,
    create_entity_mapping,
    create_relation_mapping,
    normalize_path,
)
from .utils import load_triples, tensor_to_df
from .splitting import split
from ..constants import COLUMN_TEMPORAL_LABELS
from ..inverse import relation_inverter_resolver
from ..sampling import NegativeSampler
from ..typing import (
    TEMPORAL_LABEL_HEAD,
    TEMPORAL_LABEL_RELATION,
    TEMPORAL_LABEL_TAIL,
    TEMPORAL_LABEL_TIMESTAMP,
    EntityMapping,
    LabeledQuadruples,
    MappedQuadruples,
    RelationMapping,
    TimestampMapping,
    TorchRandomHint,
)
from ..utils import format_relative_comparison

__all__ = [
    "QuadruplesFactory",
    "INVERSE_SUFFIX",
]

logger = logging.getLogger(__name__)

QUADRUPLES_DF_COLUMNS = (
    "head_id",
    "head_label",
    "relation_id",
    "relation_label",
    "tail_id",
    "tail_label",
    "timestamp_id",
    "timestamp_label",
)


def create_timestamp_mapping(timestamps: set) -> TimestampMapping:
    """Create sorted timestamp mapping."""
    # format of the timestamp is YYYY-MM-DD.
    timestamp_labels = sorted(set(timestamps), key=lambda date: datetime.strptime(date, "%Y-%m-%d"))

    return {str(label): i for (i, label) in enumerate(timestamp_labels)}


class CoreQuadruplesFactory(CoreTriplesFactory):
    """Create instances from ID-based quadruples."""

    def __init__(
        self,
        mapped_quadruples: Union[MappedQuadruples, np.ndarray],
        num_entities: int,
        num_relations: int,
        num_timestamps: int,
        create_inverse_quadruples: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        """
        Create the quadruples factory.

        :param mapped_quadruples: shape: (n, 4)
            A four-column matrix where each row are the head identifier, relation identifier, tail identifier, then timestamp.
        :param num_entities:
            The number of entities.
        :param num_relations:
            The number of relations.
        :param num_timestamps:
            The number of timestamps.
        :param create_inverse_quadruples:
            Whether to create inverse quadruples.
        :param metadata:
            Arbitrary metadata to go with the graph.

        :raises TypeError:
            if the mapped_quadruples are of non-integer dtype
        :raises ValueError:
            if the mapped_quadruples are of invalid shape
        """
        super().__init__(
            mapped_triples=mapped_quadruples[:, :3],
            num_entities=num_entities,
            num_relations=num_relations,
            create_inverse_triples=create_inverse_quadruples,
            metadata=metadata,
        )
        self.num_timestamps = num_timestamps
        # ensure torch.Tensor
        mapped_quadruples = torch.as_tensor(mapped_quadruples)
        # input validation
        if mapped_quadruples.ndim != 2 or mapped_quadruples.shape[1] != 4:
            raise ValueError(
                f"Invalid shape for mapped_quadruples: {mapped_quadruples.shape}; must be (n, 3)"
            )
        if mapped_quadruples.is_complex() or mapped_quadruples.is_floating_point():
            raise TypeError(f"Invalid type: {mapped_quadruples.dtype}. Must be integer dtype.")
        # always store as torch.long, i.e., torch's default integer dtype
        self.mapped_quadruples = mapped_quadruples.to(dtype=torch.long)

    @classmethod
    def create(
        cls,
        mapped_quadruples: MappedQuadruples,
        num_entities: Optional[int] = None,
        num_relations: Optional[int] = None,
        num_timestamps: Optional[int] = None,
        create_inverse_quadruples: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "CoreQuadruplesFactory":
        """
        Create a quadruples factory without any label information.

        :param mapped_quadruples: shape: (n, 4)
            The ID-based quadruples.
        :param num_entities:
            The number of entities. If not given, inferred from mapped_quadruples.
        :param num_relations:
            The number of relations. If not given, inferred from mapped_quadruples.
        :param num_timestamps:
            The number of timestamps. If not given, inferred from mapped_quadruples.
        :param create_inverse_quadruples:
            Whether to create inverse quadruples.
        :param metadata:
            Additional metadata to store in the factory.

        :return:
            A new quadruples factory.
        """
        if num_entities is None:
            num_entities = mapped_quadruples[:, [0, 2]].max().item() + 1
        if num_relations is None:
            num_relations = mapped_quadruples[:, 1].max().item() + 1
        if num_timestamps is None:
            num_timestamps = mapped_quadruples[:, 3].max().item() + 1
        return CoreQuadruplesFactory(
            mapped_quadruples=mapped_quadruples,
            num_entities=num_entities,
            num_relations=num_relations,
            num_timestamps=num_timestamps,
            create_inverse_quadruples=create_inverse_quadruples,
            metadata=metadata,
        )

    def __eq__(self, __o: object) -> bool:  # noqa: D105
        if not isinstance(__o, CoreQuadruplesFactory):
            return False
        return (
            (self.num_entities == __o.num_entities)
            and (self.num_relations == __o.num_relations)
            and (self.num_timestamps == __o.num_timestamps)
            and (self.num_quadruples == __o.num_quadruples)
            and (self.create_inverse_triples == __o.create_inverse_triples)
            and bool((self.mapped_quadruples == __o.mapped_quadruples).all().item())
        )

    @property
    def num_quadruples(self) -> int:  # noqa: D401
        """The number of quadruples."""
        return self.mapped_quadruples.shape[0]

    def iter_extra_repr(self) -> Iterable[str]:
        """Iterate over extra_repr components."""
        yield from super().iter_extra_repr()
        yield f"num_quadruples={self.num_quadruples}"
        for k, v in sorted(self.metadata.items()):
            if isinstance(v, (str, pathlib.Path)):
                v = f'"{v}"'
            yield f"{k}={v}"

    def with_labels(
        self,
        entity_to_id: Mapping[str, int],
        relation_to_id: Mapping[str, int],
        timestamp_to_id: Mapping[str, int],
    ) -> "QuadruplesFactory":
        """Add labeling to the QuadruplesFactory."""
        # check new label to ID mappings
        for name, columns, new_labeling in (
            ("entity", [0, 2], entity_to_id),
            ("relation", 1, relation_to_id),
            ("timestamp", 3, timestamp_to_id),
        ):
            existing_ids = set(self.mapped_quadruples[:, columns].unique().tolist())
            if not existing_ids.issubset(new_labeling.values()):
                diff = existing_ids.difference(new_labeling.values())
                raise ValueError(
                    f"Some existing IDs do not occur in the new {name} labeling: {diff}"
                )
        return QuadruplesFactory(
            mapped_quadruples=self.mapped_quadruples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            timestamp_to_id=timestamp_to_id,
            create_inverse_quadruples=self.create_inverse_quadruples,
            metadata=self.metadata,
        )

    def get_inverse_relation_id(self, relation: int) -> int:
        """Get the inverse relation identifier for the given relation."""
        if not self.create_inverse_quadruples:
            raise ValueError("Can not get inverse quadruple, they have not been created.")
        return self.relation_inverter.get_inverse_id(relation_id=relation)

    def _add_inverse_quadruples_if_necessary(
        self, mapped_quadruples: MappedQuadruples
    ) -> MappedQuadruples:
        """Add inverse quadruples if they shall be created."""
        if not self.create_inverse_quadruples:
            return mapped_quadruples

        logger.info("Creating inverse quadruples.")
        return torch.cat(
            [
                self.relation_inverter.map(batch=mapped_quadruples),
                self.relation_inverter.map(batch=mapped_quadruples, invert=True).flip(1),
            ]
        )

    def create_slcwa_instances(
        self,
        negative_sampler: HintOrType[NegativeSampler] = None,
        negative_sampler_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> SLCWAQuadrupleInstances:
        """Create sLCWA instances for this factory's quadruples."""
        cls = BatchedSLCWAInstances if negative_sampler is None else SubGraphSLCWAInstances
        if "shuffle" in kwargs:
            if kwargs.pop("shuffle"):
                warnings.warn("Training instances are always shuffled.", DeprecationWarning)
            else:
                raise AssertionError("If shuffle is provided, it must be True.")
        return cls(
            mapped_quadruples=self._add_inverse_quadruples_if_necessary(
                mapped_quadruples=self.mapped_quadruples
            ),
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            num_timestamps=self.num_timestamps,
            **kwargs,
        )

    def create_lcwa_instances(
        self, use_tqdm: Optional[bool] = None, target: Optional[int] = None
    ) -> Dataset:
        """Create LCWA instances for this factory's quadruples."""
        return LCWAQuadrupleInstances.from_quadruples(
            mapped_quadruples=self._add_inverse_quadruples_if_necessary(
                mapped_quadruples=self.mapped_quadruples
            ),
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            num_timestamps=self.num_timestamps,
            target=target,
        )

    def get_most_frequent_relations(self, n: Union[int, float]) -> Set[int]:
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

        uniq, counts = self.mapped_quadruples[:, 1].unique(return_counts=True)
        top_counts, top_ids = counts.topk(k=n, largest=True)
        return set(uniq[top_ids].tolist())

    def clone_and_exchange_quadruples(
        self,
        mapped_quadruples: MappedQuadruples,
        extra_metadata: Optional[Dict[str, Any]] = None,
        keep_metadata: bool = True,
        create_inverse_quadruples: Optional[bool] = None,
    ) -> "CoreQuadruplesFactory":
        """
        Create a new quadruples factory sharing everything except the quadruples.

        .. note ::
            We use shallow copies.

        :param mapped_quadruples:
            The new mapped quadruples.
        :param extra_metadata:
            Extra metadata to include in the new quadruples factory. If ``keep_metadata`` is true,
            the dictionaries will be unioned with precedence taken on keys from ``extra_metadata``.
        :param keep_metadata:
            Pass the current factory's metadata to the new quadruples factory
        :param create_inverse_quadruples:
            Change inverse quadruple creation flag. If None, use flag from this factory.

        :return:
            The new factory.
        """
        if create_inverse_quadruples is None:
            create_inverse_quadruples = self.create_inverse_quadruples
        return CoreQuadruplesFactory(
            mapped_quadruples=mapped_quadruples,
            num_entities=self.num_entities,
            num_relations=self.real_num_relations,
            num_timestamps=self.num_timestamps,
            create_inverse_quadruples=create_inverse_quadruples,
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
    ) -> List["CoreQuadruplesFactory"]:
        """Split a quadruples factory into a train/test.

        :param ratios:
            There are three options for this argument:

            1. A float can be given between 0 and 1.0, non-inclusive. The first set of quadruples will
               get this ratio and the second will get the rest.
            2. A list of ratios can be given for which set in which order should get what ratios as in
               ``[0.8, 0.1]``. The final ratio can be omitted because that can be calculated.
            3. All ratios can be explicitly set in order such as in ``[0.8, 0.1, 0.1]``
               where the sum of all ratios is 1.0.
        :param random_state:
            The random state used to shuffle and split the quadruples.
        :param randomize_cleanup:
            If true, uses the non-deterministic method for moving quadruples to the training set. This has the
            advantage that it does not necessarily have to move all of them, but it might be significantly
            slower since it moves one quadruple at a time.
        :param method:
            The name of the method to use, from SPLIT_METHODS. Defaults to "coverage".

        :return:
            A partition of quadruples, which are split (approximately) according to the ratios, stored QuadruplesFactory's
            which share everything else with this root quadruples factory.

        .. code-block:: python

            ratio = 0.8  # makes a [0.8, 0.2] split
            training_factory, testing_factory = factory.split(ratio)

            ratios = [0.8, 0.1]  # makes a [0.8, 0.1, 0.1] split
            training_factory, testing_factory, validation_factory = factory.split(ratios)

            ratios = [0.8, 0.1, 0.1]  # also makes a [0.8, 0.1, 0.1] split
            training_factory, testing_factory, validation_factory = factory.split(ratios)
        """
        # Make new quadruples factories for each group
        return [
            self.clone_and_exchange_quadruples(
                mapped_quadruples=quadruples,
                # do not explicitly create inverse quadruples for testing; this is handled by the evaluation code
                create_inverse_quadruples=None if i == 0 else False,
            )
            for i, quadruples in enumerate(
                split(
                    mapped_quadruples=self.mapped_quadruples,
                    ratios=ratios,
                    random_state=random_state,
                    randomize_cleanup=randomize_cleanup,
                    method=method,
                )
            )
        ]

    def entities_to_ids(self, entities: Union[Collection[int], Collection[str]]) -> Collection[int]:
        """Normalize entities to IDs.

        :param entities: A collection of either integer identifiers for entities or
            string labels for entities (that will get auto-converted)
        :returns: Integer identifiers for entities
        :raises ValueError: If the ``entities`` passed are string labels
            and this quadruples factory does not have an entity label to identifier mapping
            (e.g., it's just a base :class:`CoreQuadruplesFactory` instance)
        """
        for e in entities:
            if not isinstance(e, int):
                raise ValueError(
                    f"{self.__class__.__name__} cannot convert entity IDs from {type(e)} to int."
                )
        return cast(Collection[int], entities)

    def relations_to_ids(
        self, relations: Union[Collection[int], Collection[str]]
    ) -> Collection[int]:
        """Normalize relations to IDs.

        :param relations: A collection of either integer identifiers for relations or
            string labels for relations (that will get auto-converted)
        :returns: Integer identifiers for relations
        :raises ValueError: If the ``relations`` passed are string labels
            and this quadruples factory does not have a relation label to identifier mapping
            (e.g., it's just a base :class:`CoreQuadruplesFactory` instance)
        """
        for e in relations:
            if not isinstance(e, int):
                raise ValueError(
                    f"{self.__class__.__name__} cannot convert relation IDs from {type(e)} to int."
                )
        return cast(Collection[int], relations)

    def timestamps_to_ids(
        self, timestamps: Union[Collection[int], Collection[str]]
    ) -> Collection[int]:
        """Normalize timestamps to IDs.

        :param timestamps: A collection of either integer identifiers for timestamps or
            string labels for tiemstamps (that will get auto-converted)
        :returns: Integer identifiers for timestamps
        :raises ValueError: If the ``timestamps`` passed are string labels
            and this quadruples factory does not have a timestamp label to identifier mapping
            (e.g., it's just a base :class:`CoreQuadruplesFactory` instance)
        """
        for e in timestamps:
            if not isinstance(e, int):
                raise ValueError(
                    f"{self.__class__.__name__} cannot convert timestamp IDs from {type(e)} to int."
                )
        return cast(Collection[int], timestamps)

    def get_mask_for_relations(
        self,
        relations: Collection[int],
        invert: bool = False,
    ) -> torch.BoolTensor:
        """Get a boolean mask for quadruples with the given relations."""
        return _get_quadruple_mask(
            ids=relations,
            quadruples=self.mapped_quadruples,
            columns=1,
            invert=invert,
            max_id=self.num_relations,
        )

    def tensor_to_df(
        self,
        tensor: torch.LongTensor,
        **kwargs: Union[torch.Tensor, np.ndarray, Sequence],
    ) -> pd.DataFrame:
        """Take a tensor of quadruples and make a pandas dataframe with labels.

        :param tensor: shape: (n, 4)
            The quadruples, ID-based and in format (head_id, relation_id, tail_id, timestamp_id).
        :param kwargs:
            Any additional number of columns. Each column needs to be of shape (n,). Reserved column names:
            {"head_id", "head_label", "relation_id", "relation_label", "tail_id", "tail_label", "timestamp_id", "timestamp_label"}.
        :return:
            A dataframe with n rows, and 8 + len(kwargs) columns.
        """
        return tensor_to_df(tensor=tensor, **kwargs)

    @classmethod
    # docstr-coverage: inherited
    def from_path_binary(
        cls,
        path: Union[str, pathlib.Path, TextIO],
    ) -> "CoreQuadruplesFactory":  # noqa: D102
        """
        Load quadruples factory from a binary file.

        :param path:
            The path, pointing to an existing PyTorch .pt file.

        :return:
            The loaded quadruples factory.
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
        # load numeric quadruples
        data["mapped_quadruples"] = torch.as_tensor(
            pd.read_csv(path.joinpath(cls.quadruples_file_name), sep="\t", dtype=int).values,
            dtype=torch.long,
        )
        return data

    def to_path_binary(
        self,
        path: Union[str, pathlib.Path, TextIO],
    ) -> pathlib.Path:
        """
        Save quadruples factory to path in (PyTorch's .pt) binary format.

        :param path:
            The path to store the quadruples factory to.
        :returns:
            The path to the file that got dumped
        """
        path = normalize_path(path, mkdir=True)

        # store numeric quadruples
        pd.DataFrame(
            data=self.mapped_quadruples.numpy(),
            columns=COLUMN_TEMPORAL_LABELS,
        ).to_csv(path.joinpath(self.quadruples_file_name), sep="\t", index=False)

        # store metadata
        torch.save(self._get_binary_state(), path.joinpath(self.base_file_name))
        logger.info(f"Stored {self} to {path.as_uri()}")

        return path

    def _get_binary_state(self):
        return dict(
            num_entities=self.num_entities,
            # note: num_relations will be doubled again when instantiating with create_inverse_quadruples=True
            num_relations=self.real_num_relations,
            create_inverse_quadruples=self.create_inverse_quadruples,
            metadata=self.metadata,
        )


class QuadruplesFactory(CoreQuadruplesFactory):
    """QuadruplesFactory Class."""

    def __init__(
        self,
        mapped_quadruples: MappedQuadruples,
        entity_to_id: EntityMapping,
        relation_to_id: RelationMapping,
        timestamp_to_id: TimestampMapping,
        create_inverse_quadruples: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        num_entities: Optional[int] = None,
        num_relations: Optional[int] = None,
        num_timestamps: Optional[int] = None,
    ):  # noqa: DAR401
        """
        Initialize QuadruplesFactory.

        :param mapped_quadruples: shape: (n, 4)
            A four-column matrix where each row are the head identifier, relation identifier, tail identifier, then timestamp.
        :param entity_to_id:
            The mapping from entities' labels to their indices.
        :param relation_to_id:
            The mapping from relations' labels to their indices.
        :param timestamp_to_id:
            The mapping from timestamps' labels to their indices.
        :param create_inverse_quadruples:
            Whether to create inverse quadruples.
        :param metadata:
            Arbitrary metadata to go with the graph.
        :param num_entities:
            The number of entities.
        :param num_relations:
            The number of relations.
        :param num_timestamps:
            The number of timestamps.
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
        self.timestamp_labeling = Labeling(label_to_id=timestamp_to_id)
        if num_timestamps is None:
            num_timestamps = self.timestamp_labeling.max_id
        elif num_timestamps != self.timestamp_labeling.max_id:
            raise ValueError(
                f"Mismatch between the number of timestamps in labeling ({self.timestamp_labeling.max_id}) "
                f"vs. explicitly provided num_timestamps={num_timestamps}",
            )
        super().__init__(
            mapped_quadruples=mapped_quadruples,
            num_entities=num_entities,
            num_relations=num_relations,
            num_timestamps=num_timestamps,
            create_inverse_quadruples=create_inverse_quadruples,
            metadata=metadata,
        )
        self.create_inverse_quadruples = create_inverse_quadruples
        self.mapped_quadruples = mapped_quadruples
        self.num_timestamps = len(timestamp_to_id)
        # self.timestamp_to_id = timestamp_to_id
        # self.timestamp_labeling = Labeling(label_to_id=timestamp_to_id)

    @classmethod
    def from_labeled_quadruples(
        cls,
        quadruples: LabeledQuadruples,
        create_inverse_quadruples: bool = False,
        entity_to_id: Optional[EntityMapping] = None,
        relation_to_id: Optional[RelationMapping] = None,
        timestamp_to_id: Optional[TimestampMapping] = None,
        compact_id: bool = True,
        filter_out_candidate_inverse_relations: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "QuadruplesFactory":
        """Create QuadruplesFactory from original data, i.e. with the real labels.

        :param quadruples: shape: (n, 4), dtype: str
            The label-based quadruples.
        :param create_inverse_quadruples:
            Whether to create inverse quadruples.
        :param entity_to_id:
            The mapping from entity labels to ID. If None, create a new one from the quadruples.
        :param relation_to_id:
            The mapping from relations labels to ID. If None, create a new one from the quadruples.
        :param timestamp_to_id:
            The mapping from timestamp labels to ID. If None, create a new one from the quadruples.
        :param compact_id:
            Whether to compact IDs such that the IDs are consecutive.
        :param filter_out_candidate_inverse_relations:
            Whether to remove quadruples with relations with the inverse suffix.
        :param metadata:
            Arbitrary key/value pairs to store as metadata.

        :return:
            A new quadruples factory.

        """
        # Check if the quadruples are inverted already
        # We re-create them pure index based to ensure that _all_ inverse quadruples are present and that they are
        # contained if and only if create_inverse_quadruples is True.
        if filter_out_candidate_inverse_relations:
            unique_relations, inverse = np.unique(quadruples[:, 1], return_inverse=True)
            suspected_to_be_inverse_relations = {
                r for r in unique_relations if r.endswith(INVERSE_SUFFIX)
            }
            if len(suspected_to_be_inverse_relations) > 0:
                logger.warning(
                    f"Some quadruples already have the inverse relation suffix {INVERSE_SUFFIX}. "
                    f"Re-creating inverse quadruples to ensure consistency. You may disable this behaviour by passing "
                    f"filter_out_candidate_inverse_relations=False",
                )
                relation_ids_to_remove = [
                    i
                    for i, r in enumerate(unique_relations.tolist())
                    if r in suspected_to_be_inverse_relations
                ]
                mask = np.isin(element=inverse, test_elements=relation_ids_to_remove, invert=True)
                logger.info(f"keeping {mask.sum() / mask.shape[0]} quadruples.")
                quadruples = quadruples[mask]

        # Generate entity mapping if necessary
        if entity_to_id is None:
            entity_to_id = create_entity_mapping(triples=quadruples)
        if compact_id:
            entity_to_id = compact_mapping(mapping=entity_to_id)[0]

        # Generate relation mapping if necessary
        if relation_to_id is None:
            relation_to_id = create_relation_mapping(quadruples[:, 1])
        if compact_id:
            relation_to_id = compact_mapping(mapping=relation_to_id)[0]

        # Generate timestamp mapping if necessary
        if timestamp_to_id is None:
            timestamp_to_id = create_timestamp_mapping(quadruples[:, 3])

        # Map quadruples of labels to quadruples of IDs
        mapped_quadruples = _map_triples_elements_to_ids(
            quadruples=quadruples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            timestamp_to_id=timestamp_to_id,
        )

        return cls(
            mapped_quadruples=mapped_quadruples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            timestamp_to_id=timestamp_to_id,
            create_inverse_quadruples=create_inverse_quadruples,
            metadata=metadata,
        )

    @classmethod
    def from_path(
        cls,
        path: Union[str, pathlib.Path],
        create_inverse_quadruples: bool = False,
        entity_to_id: Optional[EntityMapping] = None,
        relation_to_id: Optional[RelationMapping] = None,
        timestamp_to_id: Optional[TimestampMapping] = None,
        compact_id: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        load_quadruples_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> "QuadruplesFactory":
        """Create QuadruplesFactory from dataset Path."""
        path = normalize_path(path)
        quadruples = load_triples(path, **(load_quadruples_kwargs or {}))
        return cls.from_labeled_quadruples(
            quadruples=quadruples,
            create_inverse_quadruples=create_inverse_quadruples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            timestamp_to_id=timestamp_to_id,
            compact_id=compact_id,
            metadata={
                "path": path,
                **(metadata or {}),
            },
        )

    def to_path_binary(
        self,
        path: Union[str, pathlib.Path, TextIO],
    ) -> pathlib.Path:
        """
        Save quadruples factory to path in (PyTorch's .pt) binary format.

        :param path:
            The path to store the quadruple factory to.
        :return:
            The path to store the dataset
        """
        path = normalize_path(path)
        path.mkdir(exist_ok=True, parents=True)  # type: ignore

        # store numeric quadruples
        pd.DataFrame(
            data=self.mapped_quadruples.numpy(),
            columns=[TEMPORAL_LABEL_HEAD, TEMPORAL_LABEL_RELATION, TEMPORAL_LABEL_TAIL, TEMPORAL_LABEL_TIMESTAMP],
        ).to_csv(
            path.joinpath(self.quadruples_file_name), sep="\t", index=False  # type: ignore
        )

        # store metadata
        torch.save(self._get_binary_state(), path.joinpath(self.base_file_name))  # type: ignore
        logger.info(f"Stored {self} to {path.as_uri()}")  # type: ignore

        return path  # type: ignore

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
    def timestamp_to_id(self) -> Mapping[str, int]:
        """Return the mapping from timestamps labels to IDs."""
        return self.timestamp_labeling.label_to_id

    @property
    def timestamp_id_to_label(self) -> Mapping[int, str]:
        """Return the mapping from timestamps IDs to labels."""
        return self.timestamp_labeling.id_to_label

    @property
    def quadruples(self) -> np.ndarray:  # noqa: D401
        """The labeled quadruples, a 4-column matrix where each row are the head label, relation label, tail label and timestamp label."""
        logger.warning(
            "Reconstructing all label-based quadruples. This is expensive and rarely needed."
        )
        return self.label_quadruples(self.mapped_quadruples)

    def label_quadruples(
        self,
        quadruples: MappedQuadruples,
        unknown_entity_label: str = "[UNKNOWN]",
        unknown_relation_label: Optional[str] = None,
        unknown_timestamp_label: Optional[str] = None,
    ) -> LabeledQuadruples:
        """
        Convert ID-based quadruples to label-based ones.

        :param quadruples:
            The ID-based quadruples.
        :param unknown_entity_label:
            The label to use for unknown entity IDs.
        :param unknown_relation_label:
            The label to use for unknown relation IDs.
        :param unknown_timestamp_label:
            The label to use for unknown timestamp IDs.
        :return:
            The same triples, but labeled.
        """
        if len(quadruples) == 0:
            return np.empty(shape=(0, 4), dtype=str)
        if unknown_relation_label is None:
            unknown_relation_label = unknown_entity_label
        if unknown_timestamp_label is None:
            unknown_timestamp_label = unknown_entity_label
        return np.stack(
            [
                labeling.label(ids=column, unknown_label=unknown_label)
                for (labeling, unknown_label), column in zip(
                    [
                        (self.entity_labeling, unknown_entity_label),
                        (self.relation_labeling, unknown_relation_label),
                        (self.entity_labeling, unknown_entity_label),
                        (self.timestamp_labeling, unknown_timestamp_label),
                    ],
                    quadruples.t().numpy(),
                )
            ],
            axis=1,
        )

    def entities_to_ids(
        self, entities: Union[Collection[int], Collection[str]]
    ) -> Collection[int]:  # noqa: D102
        return _ensure_ids(labels_or_ids=entities, label_to_id=self.entity_labeling.label_to_id)

    def relations_to_ids(
        self, relations: Union[Collection[int], Collection[str]]
    ) -> Collection[int]:  # noqa: D102
        return _ensure_ids(labels_or_ids=relations, label_to_id=self.relation_labeling.label_to_id)

    def timestamps_to_ids(
        self, timestamps: Union[Collection[int], Collection[str]]
    ) -> Collection[int]:  # noqa: D102
        return _ensure_ids(
            labels_or_ids=timestamps, label_to_id=self.timestamp_labeling.label_to_id
        )

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
            timestamp=self.timestamp_labeling,
        ).items():
            assert labeling is not None
            data[f"{column}_label"] = labeling.label(
                ids=data[f"{column}_id"],
                unknown_label=("[unknown_" + column + "]").upper(),
            )

        # Re-order columns
        columns = list(QUADRUPLES_DF_COLUMNS) + old_col[4:]
        return data.loc[:, columns]

    def new_with_restriction(
        self,
        entities: Union[None, Collection[int], Collection[str]] = None,
        relations: Union[None, Collection[int], Collection[str]] = None,
        timestamps: Union[None, Collection[int], Collection[str]] = None,
        invert_entity_selection: bool = False,
        invert_relation_selection: bool = False,
        invert_timestamp_selection: bool = False,
    ) -> "QuadruplesFactory":  # noqa: D102
        if entities is None and relations is None and timestamps is None:
            return self
        if entities is not None:
            entities = self.entities_to_ids(entities=entities)
        if relations is not None:
            relations = self.relations_to_ids(relations=relations)
        if timestamps is not None:
            timestamps = self.timestamps_to_ids(timestamps=timestamps)
        return (
            super()
            .new_with_restriction(
                entities=entities,
                relations=relations,
                timestamps=timestamps,
                invert_entity_selection=invert_entity_selection,
                invert_relation_selection=invert_relation_selection,
                invert_timestamp_selection=invert_timestamp_selection,
            )
            .with_labels(
                entity_to_id=self.entity_to_id,
                relation_to_id=self.relation_to_id,
                timestamp_to_id=self.timestamp_to_id,
            )
        )

    def map_quadruples(self, quadruples: LabeledQuadruples) -> MappedQuadruples:
        """Convert label-based quadruples to ID-based quadruples."""
        return _map_triples_elements_to_ids(
            quadruples=quadruples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            timestamp_to_id=self.timestamp_to_id,
        )

    def create_slcwa_instances(
        self,
        negative_sampler: HintOrType[NegativeSampler] = None,
        negative_sampler_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> SLCWAQuadrupleInstances:
        """Create SLCWA instances for this factory's quadruples."""
        return SLCWAQuadrupleInstances.from_quadruples(
            mapped_quadruples=self._add_inverse_quadruples_if_necessary(
                mapped_quadruples=self.mapped_quadruples
            ),
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            negative_sampler=negative_sampler,
            negative_sampler_kwargs=negative_sampler_kwargs,
        )

    def _add_inverse_quadruples_if_necessary(
        self, mapped_quadruples: MappedQuadruples
    ) -> MappedQuadruples:
        if not self.create_inverse_quadruples:
            return self.mapped_quadruples
        logger.info("Creating inverse quadruples")
        # relation_inverter: inherited from TriplesFactory
        inverted = self.relation_inverter.map(batch=mapped_quadruples, invert=True)
        # switch the positions of head and tail
        inverted[:, [0, 1, 2, 3]] = inverted[:, [2, 1, 0, 3]]
        return torch.LongTensor(
            torch.cat(
                [self.relation_inverter.map(batch=mapped_quadruples), inverted],
            )
        )
