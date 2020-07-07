# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch.utils import data

from ..typing import EntityMapping, MappedTriples, RelationMapping
from ..utils import fix_dataclass_init_docs

__all__ = [
    'Instances',
    'SLCWAInstances',
    'LCWAInstances',
    'MultimodalInstances',
    'MultimodalSLCWAInstances',
    'MultimodalLCWAInstances',
]


@fix_dataclass_init_docs
@dataclass
class Instances(data.Dataset):
    """Triples and mappings to their indices."""

    #: A PyTorch tensor of triples
    mapped_triples: MappedTriples

    #: A mapping from relation labels to integer identifiers
    entity_to_id: EntityMapping

    #: A mapping from relation labels to integer identifiers
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


@fix_dataclass_init_docs
@dataclass
class SLCWAInstances(Instances):
    """Triples and mappings to their indices for sLCWA."""

    def __getitem__(self, item):  # noqa: D105
        return self.mapped_triples[item]


@fix_dataclass_init_docs
@dataclass
class LCWAInstances(Instances):
    """Triples and mappings to their indices for LCWA."""

    labels: np.ndarray

    def __getitem__(self, item):  # noqa: D105
        # Create dense target
        batch_labels_full = torch.zeros(self.num_entities)
        batch_labels_full[self.labels[item]] = 1
        return self.mapped_triples[item], batch_labels_full


@fix_dataclass_init_docs
@dataclass
class MultimodalInstances(Instances):
    """Triples and mappings to their indices as well as multimodal data."""

    numeric_literals: Mapping[str, np.ndarray]
    literals_to_id: Mapping[str, int]


@fix_dataclass_init_docs
@dataclass
class MultimodalSLCWAInstances(SLCWAInstances, MultimodalInstances):
    """Triples and mappings to their indices as well as multimodal data for sLCWA."""


@fix_dataclass_init_docs
@dataclass
class MultimodalLCWAInstances(LCWAInstances, MultimodalInstances):
    """Triples and mappings to their indices as well as multimodal data for LCWA."""
