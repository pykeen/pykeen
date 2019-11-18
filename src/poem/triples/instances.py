# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch.utils import data

from .utils import Assumption
from ..typing import EntityMapping, MappedTriples, RelationMapping

__all__ = [
    'Instances',
    'OWAInstances',
    'LCWAInstances',
    'MultimodalInstances',
    'MultimodalOWAInstances',
    'MultimodalLCWAInstances',
]


@dataclass
class Instances(data.Dataset):
    """Triples and mappings to their indices."""

    mapped_triples: MappedTriples
    entity_to_id: EntityMapping
    relation_to_id: RelationMapping

    @property
    def num_instances(self) -> int:  # noqa: D401
        """The number of instances."""
        return self.mapped_triples.shape[0]

    @property
    def num_entities(self) -> int:  # noqa: D401
        """The number of entities."""
        return len(self.entity_to_id)

    def __len__(self):  # noqa: D105
        return self.num_instances


@dataclass
class OWAInstances(Instances):
    """Triples and mappings to their indices for OWA."""

    assumption: Assumption = Assumption.open

    def __getitem__(self, item):  # noqa: D105
        return self.mapped_triples[item]


@dataclass
class LCWAInstances(Instances):
    """Triples and mappings to their indices for LCWA."""

    labels: np.ndarray
    assumption: Assumption = Assumption.local_closed

    def __getitem__(self, item):  # noqa: D105
        # Create dense target
        batch_labels_full = torch.zeros(self.num_entities)
        batch_labels_full[self.labels[item]] = 1
        return self.mapped_triples[item], batch_labels_full


@dataclass
class MultimodalInstances(Instances):
    """Triples and mappings to their indices as well as multimodal data."""

    numeric_literals: Mapping[str, np.ndarray]
    literals_to_id: Mapping[str, int]


@dataclass
class MultimodalOWAInstances(OWAInstances, MultimodalInstances):
    """Triples and mappings to their indices as well as multimodal data for OWA."""


@dataclass
class MultimodalLCWAInstances(LCWAInstances, MultimodalInstances):
    """Triples and mappings to their indices as well as multimodal data for LCWA."""
