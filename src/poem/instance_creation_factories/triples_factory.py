# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

import logging
import os
import timeit
from typing import Optional, TextIO, Union

import numpy as np

from .instances import CWAInstances, OWAInstances
from ..preprocessing.instance_creation_utils import create_multi_label_objects_instance
from ..preprocessing.utils import (
    create_entity_and_relation_mappings, load_triples, map_triples_elements_to_ids,
)

__all__ = [
    'TriplesFactory',
]

log = logging.getLogger(__name__)


class TriplesFactory:
    """Create instances given the path to triples."""

    def __init__(
            self,
            *,
            path: Union[None, str, TextIO] = None,
            triples: Optional[np.ndarray] = None,
            create_inverse_triples: bool = False,
    ) -> None:
        """Initialize the triples factory.

        :param path:
        :param triples:
        :param create_inverse_triples:
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

        self.entity_to_id, self.relation_to_id = create_entity_and_relation_mappings(self.triples)
        self.mapped_triples = map_triples_elements_to_ids(
            triples=self.triples,
            entity_to_id=self.entity_to_id,
            rel_to_id=self.relation_to_id,
        )
        self.all_entities = np.array(list(self.entity_to_id.values()))
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

    def __repr__(self):  # noqa: D105
        return f'{self.__class__.__name__}(path="{self.path}")'

    def _create_inverse_triples(self) -> None:
        start = timeit.default_timer()
        log.info(f'Creating inverse triples')
        inverse_triples = np.flip(self.mapped_triples.copy())
        inverse_triples[:, 1:2] += self.num_relations
        self.mapped_triples = np.concatenate((self.mapped_triples, inverse_triples))
        log.info(f'Created inverse triples. It took {timeit.default_timer() - start:.2f} seconds')
        # The number of relations has to be doubled when using inverse triples
        # FIXME need to update self.relation_to_id dictionary!
        self.num_relations = self.num_relations * 2

    def create_owa_instances(self) -> OWAInstances:
        """Create OWA instances for this factory's triples."""
        return OWAInstances(
            instances=self.mapped_triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
        )

    def create_cwa_instances(self) -> CWAInstances:
        """Create CWA instances for this factory's triples."""
        s_r_to_mulit_objects = create_multi_label_objects_instance(
            triples=self.mapped_triples,
        )

        subject_relation_pairs = np.array(list(s_r_to_mulit_objects.keys()), dtype=np.float)
        labels = np.array(list(s_r_to_mulit_objects.values()))

        return CWAInstances(
            instances=subject_relation_pairs,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            labels=labels,
        )

    def map_triples_to_id(self, path_to_triples: str) -> np.array:
        """Load triples and map to ids based on the existing id mappings of the triples factory."""
        triples = load_triples(path_to_triples)
        return map_triples_elements_to_ids(
            triples=triples,
            entity_to_id=self.entity_to_id,
            rel_to_id=self.relation_to_id,
        )
