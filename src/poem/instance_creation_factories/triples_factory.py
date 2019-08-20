# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

import logging
import os
import timeit
from collections import defaultdict
from typing import Dict, Optional, TextIO, Tuple, Union

import numpy as np
from tqdm import tqdm

from .instances import CWAInstances, OWAInstances
from .utils import load_triples
from ..typing import EntityMapping, LabeledTriples, MappedTriples, RelationMapping
from ..utils import slice_triples

__all__ = [
    'TriplesFactory',
]

log = logging.getLogger(__name__)


def _create_multi_label_objects_instance(triples: np.array) -> Dict[tuple, np.array]:
    """Create for each (s,r) pair the multi object label."""
    log.info(f'Creating multi label objects instance')

    s_r_to_multi_objects_new = _create_multi_label_instances(
        triples,
        element_1_index=0,
        element_2_index=1,
        label_index=2,
    )

    log.info(f'Created multi label objects instance')

    return s_r_to_multi_objects_new


def _create_multi_label_instances(
    triples: np.array,
    element_1_index: int,
    element_2_index: int,
    label_index: int,
) -> Dict[tuple, np.array]:
    """Create for each (element_1, element_2) pair the multi-label."""
    instance_to_multi_label = defaultdict(set)
    for row in tqdm(triples):
        instance_to_multi_label[(row[element_1_index], row[element_2_index])].add(row[label_index])

    # Create lists out of sets for proper numpy indexing when loading the labels
    instance_to_multi_label_new = {
        key: list(value)
        for key, value in instance_to_multi_label.items()
    }

    return instance_to_multi_label_new


def _create_entity_and_relation_mappings(
    triples: np.array
) -> Tuple[np.ndarray, EntityMapping, np.ndarray, RelationMapping]:
    """Map entities and relations to ids."""
    subjects, relations, objects = triples[:, 0], triples[:, 1], triples[:, 2]

    # Sorting ensures consistent results when the triples are permuted
    entity_labels = sorted(set(subjects).union(objects))
    relation_labels = sorted(set(relations))

    entity_ids = np.arange(len(entity_labels))
    entity_label_to_id = dict(zip(entity_labels, entity_ids))

    relation_ids = np.arange(len(entity_labels))
    relation_label_to_id = dict(zip(relation_labels, relation_ids))

    return (
        entity_ids,
        entity_label_to_id,
        relation_ids,
        relation_label_to_id,
    )


def _map_triples_elements_to_ids(
    triples: LabeledTriples,
    entity_to_id: EntityMapping,
    relation_to_id: RelationMapping,
) -> np.ndarray:
    """Map entities and relations to pre-defined ids."""
    heads, relations, tails = slice_triples(triples)

    # When triples that don't exist are trying to be mapped, they get the id "-1"
    subject_column = np.vectorize(entity_to_id.get)(heads, [-1])
    relation_column = np.vectorize(relation_to_id.get)(relations, [-1])
    object_column = np.vectorize(entity_to_id.get)(tails, [-1])

    # Filter all non-existent triples
    subject_filter = subject_column < 0
    relation_filter = relation_column < 0
    object_filter = object_column < 0
    num_no_subject = subject_filter.sum()
    num_no_relation = relation_filter.sum()
    num_no_object = object_filter.sum()

    if (num_no_subject > 0) or (num_no_relation > 0) or (num_no_object > 0):
        log.warning(
            "You're trying to map triples with entities and/or relations that are not in the training set."
            "These triples will be excluded from the mapping")
        non_mappable_triples = (subject_filter | relation_filter | object_filter)
        subject_column = subject_column[~non_mappable_triples, None]
        relation_column = relation_column[~non_mappable_triples, None]
        object_column = object_column[~non_mappable_triples, None]
        log.warning(f"In total {non_mappable_triples.sum():.0f} from {triples.shape[0]:.0f} triples were filtered out")

    triples_of_ids = np.concatenate([subject_column, relation_column, object_column], axis=1)

    triples_of_ids = np.array(triples_of_ids, dtype=np.long)
    # Note: Unique changes the order of the triples
    # Note: Using unique means implicit balancing of training samples
    return np.unique(ar=triples_of_ids, axis=0)


class TriplesFactory:
    """Create instances given the path to triples."""

    #: The integer indexes for each entity
    all_entities: np.ndarray

    #: The mapping from entities' labels to their indexes
    entity_to_id: EntityMapping

    #: The integer indexes for each relation
    all_relations: np.ndarray

    #: The mapping from relations' labels to their indexes
    relation_to_id: RelationMapping

    #: A three-column matrix where each row are the subject label,
    #: relation label, then object label
    triples: LabeledTriples

    #: A three-column matrix where each row are the subject identifier,
    #: relation identifier, then object identifier
    mapped_triples: MappedTriples

    def __init__(
        self,
        *,
        path: Union[None, str, TextIO] = None,
        triples: Optional[LabeledTriples] = None,
        create_inverse_triples: bool = False,
    ) -> None:
        """Initialize the triples factory.

        :param path: The path to a 3-column TSV file with triples in it. If not specified,
         you should specify ``triples``.
        :param triples:  A 3-column numpy array with triples in it. If not specified,
         you should specify ``path``
        :param create_inverse_triples: Should inverse triples be created? Defaults to False.
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

        (
            self.all_entities,
            self.entity_to_id,
            self.all_relations,
            self.relation_to_id,
        ) = _create_entity_and_relation_mappings(self.triples)

        self.mapped_triples = _map_triples_elements_to_ids(
            triples=self.triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
        )

        self.create_inverse_triples = create_inverse_triples
        # This creates inverse triples and appends them to the mapped triples
        if self.create_inverse_triples:
            self._create_inverse_triples()

    @property
    def num_entities(self) -> int:  # noqa: D401
        """The number of unique entities."""
        return len(self.entity_to_id)

    @property
    def num_relations(self) -> int:  # noqa: D401
        """The number of unique relations."""
        return len(self.relation_to_id)

    def get_inverse_relation_id(self, relation: str) -> int:
        """Get the inverse relation identifier for the given relation."""
        if not self.create_inverse_triples:
            raise ValueError('Can not get inverse triple, they have not been created.')
        return self.relation_to_id[relation] + self.num_relations // 2

    def __repr__(self):  # noqa: D105
        return f'{self.__class__.__name__}(path="{self.path}")'

    def _create_inverse_triples(self) -> None:
        start = timeit.default_timer()
        log.info(f'Creating inverse triples')
        inverse_triples = np.flip(self.mapped_triples.copy())
        inverse_triples[:, 1:2] += self.num_relations
        self.mapped_triples = np.concatenate((self.mapped_triples, inverse_triples))
        log.info(f'Created inverse triples. It took {timeit.default_timer() - start:.2f} seconds')
        # The newly added inverse relations have to be added to the relation_id mapping
        inverse_triples_to_id_mapping = {
            f"{relation}_inverse": relation_id + self.num_relations
            for relation, relation_id in self.relation_to_id.items()
        }
        self.relation_to_id.update(inverse_triples_to_id_mapping)

    def create_owa_instances(self) -> OWAInstances:
        """Create OWA instances for this factory's triples."""
        return OWAInstances(
            mapped_triples=self.mapped_triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
        )

    def create_cwa_instances(self) -> CWAInstances:
        """Create CWA instances for this factory's triples."""
        s_r_to_multi_objects = _create_multi_label_objects_instance(
            triples=self.mapped_triples,
        )

        subject_relation_pairs = np.array(list(s_r_to_multi_objects.keys()), dtype=np.float)
        labels = np.array(list(s_r_to_multi_objects.values()))

        return CWAInstances(
            mapped_triples=subject_relation_pairs,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            labels=labels,
        )
