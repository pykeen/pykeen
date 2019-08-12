# -*- coding: utf-8 -*-

"""Training loops for KGE models using multi-modal information."""

from abc import ABC, abstractmethod
from typing import List, Mapping, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..instance_creation_factories import Instances, TriplesFactory
from ..models.base import BaseModule
from ..version import get_version

__all__ = [
    'TrainingLoop',
]


class TrainingLoop(ABC):
    """A training loop."""

    def __init__(
            self,
            optimizer: Optional[torch.optim.Optimizer] = None,
            model: Optional[BaseModule] = None,
    ) -> None:
        """Initialize the training loop."""
        self.model = model
        self.optimizer = optimizer
        self.losses_per_epochs = []

    @property
    def triples_factory(self) -> TriplesFactory:  # noqa: D401
        """The triples factory in the model."""
        return self.model.triples_factory

    @property
    def all_entities(self) -> np.ndarray:  # noqa: D401
        """All entities in the triples factory."""
        return self.model.triples_factory.all_entities

    @property
    def device(self):  # noqa: D401
        """The device used by the model."""
        return self.model.device

    @abstractmethod
    def train(
            self,
            training_instances: Instances,
            num_epochs: int,
            batch_size: int,
    ) -> List[float]:
        """Train the KGE model.

        :return: A pair of the KGE model and the losses per epoch.
        """
        raise NotImplementedError

    def _get_entity_to_vector_dict(self) -> Mapping:
        raise NotImplementedError

    def to_embeddingdb(self, session=None, use_tqdm: bool = False):
        """Upload to the embedding database.

        :param session: Optional SQLAlchemy session
        :param use_tqdm: Use :mod:`tqdm` progress bar?
        :rtype: embeddingdb.sql.models.Collection
        """
        from embeddingdb.sql.models import Embedding, Collection

        if session is None:
            from embeddingdb.sql.models import get_session
            session = get_session()

        collection = Collection(
            package_name='poem',
            package_version=get_version(),
            dimensions=...,
            extras=...,
        )

        it = self._get_entity_to_vector_dict().items()
        if use_tqdm:
            it = tqdm(it, desc='Building SQLAlchemy models')
        for curie, vector in it:
            embedding = Embedding(
                collection=collection,
                curie=curie,
                vector=vector,
            )
            session.add(embedding)
        session.add(collection)
        session.commit()
        return collection
