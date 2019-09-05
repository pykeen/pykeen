# -*- coding: utf-8 -*-

"""Base module for all KGE models."""

import random
from abc import abstractmethod
from collections import defaultdict
from typing import ClassVar, Dict, Iterable, Optional, Set, Union

import click
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .cli import build_cli_from_cls
from ..instance_creation_factories import TriplesFactory
from ..typing import OptionalLoss
from ..utils import resolve_device
from ..version import get_version

__all__ = [
    'BaseModule',
    'RegularizedModel',
]


class BaseModule(nn.Module):
    """A base module for all of the KGE models."""

    #: A dictionary of hyper-parameters to the models that use them
    _hyperparameter_usage: ClassVar[Dict[str, Set[str]]] = defaultdict(set)

    #: The command line interface for this model
    cli: ClassVar[click.Command]

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        entity_embeddings: nn.Embedding = None,
        criterion: OptionalLoss = None,
        predict_with_sigmoid: bool = False,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the module."""
        super().__init__()

        # Initialize the device
        self._set_device(preferred_device)

        self.random_seed = random_seed

        # Random seeds have to set before the embeddings are initialized
        if self.random_seed is not None:
            np.random.seed(seed=self.random_seed)
            torch.manual_seed(seed=self.random_seed)
            random.seed(self.random_seed)

        # Loss
        if criterion is None:
            self.criterion = nn.MarginRankingLoss()
        else:
            self.criterion = criterion

        # TODO: Check loss functions that require 1 and -1 as label but only
        self.is_mr_loss = isinstance(self.criterion, nn.MarginRankingLoss)

        # The triples factory facilitates access to the dataset.
        self.triples_factory = triples_factory

        #: The dimension of the embeddings to generate
        self.embedding_dim = embedding_dim

        # The embeddings are first initiated when calling the fit function
        self.entity_embeddings = entity_embeddings

        # Marker to check whether the forward constraints of a models has been applied before starting loss calculation
        self.forward_constraint_applied = False

        '''
        When predict_with_sigmoid is set to True, the sigmoid function is applied to the logits during evaluation and
        also for predictions after training, but has no effect on the training.
        '''
        self.predict_with_sigmoid = predict_with_sigmoid

    def __init_subclass__(cls, **kwargs):
        """Initialize the subclass while keeping track of hyper-parameters."""
        super().__init_subclass__(**kwargs)

        # Keep track of the hyper-parameters that are used across all
        # subclasses of BaseModule
        for k, v in cls.__init__.__annotations__.items():
            if k not in {'return', 'triples_factory'}:
                BaseModule._hyperparameter_usage[k].add(cls.__name__)

        cls.cli = build_cli_from_cls(cls)

    @property
    def num_entities(self) -> int:  # noqa: D401
        """The number of entities in the knowledge graph."""
        return self.triples_factory.num_entities

    @property
    def num_relations(self) -> int:  # noqa: D401
        """The number of unique relation types in the knowledge graph."""
        return self.triples_factory.num_relations

    def _set_device(self, device: Union[None, str, torch.device] = None) -> None:
        """Set the Torch device to use."""
        self.device = resolve_device(device=device)

    def to_device_(self) -> 'BaseModule':
        """Transfer model to device."""
        self.to(self.device)
        torch.cuda.empty_cache()
        return self

    def to_cpu_(self) -> 'BaseModule':
        """Transfer the entire model to CPU."""
        self._set_device('cpu')
        return self.to_device_()

    def to_gpu_(self) -> 'BaseModule':
        """Transfer the entire model to GPU."""
        self._set_device('cuda')
        return self.to_device_()

    def predict_scores(self, triples: torch.LongTensor) -> torch.FloatTensor:
        """Calculate the scores for triples.

        This method takes subject, relation and object of each triple and calculates the corresponding score.

        Additionally, the model is set to evaluation mode.

        :param triples: torch.Tensor, shape: (number of triples, 3), dtype: long
            The indices of (subject, relation, object) triples.

        :return: torch.Tensor, shape: (number of triples, 1), dtype: float
            The score for each triple.
        """
        # Enforce evaluation mode
        self.eval()

        scores = self.forward_owa(triples)
        if self.predict_with_sigmoid:
            scores = torch.sigmoid(scores)
        return scores

    def predict_scores_all_objects(self, batch: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass using right side (object) prediction for obtaining scores of all possible objects.

        This method calculates the score for all possible objects for each (subject, relation) pair.

        Additionally, the model is set to evaluation mode.

        :param batch: torch.Tensor, shape: (batch_size, 2), dtype: long
            The indices of (subject, relation) pairs.

        :return: torch.Tensor, shape: (batch_size, num_entities), dtype: float
            For each s-p pair, the scores for all possible objects.
        """
        # Enforce evaluation mode
        self.eval()

        scores = self.forward_cwa(batch)
        if self.predict_with_sigmoid:
            scores = torch.sigmoid(scores)
        return scores

    def predict_scores_all_subjects(
        self,
        batch: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Forward pass using left side (subject) prediction for obtaining scores of all possible subjects.

        This method calculates the score for all possible subjects for each (relation, object) pair.

        Additionally, the model is set to evaluation mode.

        :param batch: torch.Tensor, shape: (batch_size, 2), dtype: long
            The indices of (relation, object) pairs.

        :return: torch.Tensor, shape: (batch_size, num_entities), dtype: float
            For each p-o pair, the scores for all possible subjects.
        """
        # Enforce evaluation mode
        self.eval()

        '''
        In case the model was trained using inverse triples, the scoring of all subjects is not handled by calculating
        the scores for all subjects based on a (relation, object) pair, but instead all possible objects are calculated
        for a (object, inverse_relation) pair.
        '''
        if not self.triples_factory.create_inverse_triples:
            scores = self.forward_inverse_cwa(batch)
            if self.predict_with_sigmoid:
                scores = torch.sigmoid(scores)
            return scores

        '''
        The POEM package handles _inverse relations_ by adding the number of relations to the index of the
        _native relation_.
        Example:
        The triples/knowledge graph used to train the model contained 100 relations. Due to using inverse relations,
        the model now has an additional 100 inverse relations. If the _native relation_ has the index 3, the index
        of the _inverse relation_ is 103.
        '''
        # The number of relations stored in the triples factory includes the number of inverse relations
        num_relations = self.triples_factory.num_relations // 2
        batch[:, 0] = batch[:, 0] + num_relations

        # The forward cwa function requires (entity, relation) pairs instead of (relation, entity)
        batch = batch.flip(1)
        scores = self.forward_cwa(batch)
        if self.predict_with_sigmoid:
            scores = torch.sigmoid(scores)
        return scores

    def compute_mr_loss(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute the mean ranking loss for the positive and negative scores.

        :param positive_scores: torch.Tensor, shape: s, dtype: float
            The scores for positive triples.
        :param negative_scores: torch.Tensor, shape: s, dtype: float
            The scores for negative triples.

        :return: torch.Tensor, dtype: float, scalar
            The margin ranking loss value.
        """
        assert self.is_mr_loss, 'The chosen criterion does not allow the calculation of margin ranking losses. ' \
                                'Please use the compute_label_loss method instead.'
        y = torch.ones_like(negative_scores, device=self.device)
        loss = self.criterion(positive_scores, negative_scores, y)
        return loss

    def compute_label_loss(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute the classification loss.

        :param predictions: shape: s
            The predicted scores.
        :param labels: shape: s
            The target values.

        :return: torch.Tensor, dtype: float, scalar
            The label loss value.
        """
        assert not self.is_mr_loss, 'The chosen criterion does not allow the calculation of margin label losses. ' \
                                    'Please use the compute_mr_loss method instead.'
        loss = self.criterion(predictions, labels)
        return loss

    @abstractmethod
    def forward_owa(self, batch: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass for training with the OWA.

        This method takes subject, relation and object of each triple and calculates the corresponding score.

        :param batch: torch.Tensor, shape: (batch_size, 3), dtype: long
            The indices of (subject, relation, object) triples.

        :return: torch.Tensor, shape: (batch_size, 1), dtype: float
            The score for each triple.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass using right side (object) prediction for training with the CWA.

        This method calculates the score for all possible objects for each (subject, relation) pair.

        :param batch: torch.Tensor, shape: (batch_size, 2), dtype: long
            The indices of (subject, relation) pairs.

        :return: torch.Tensor, shape: (batch_size, num_entities), dtype: float
            For each s-p pair, the scores for all possible objects.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_inverse_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass using left side (subject) prediction for training with the CWA.

        This method calculates the score for all possible subjects for each (relation, object) pair.

        :param batch: torch.Tensor, shape: (batch_size, 2), dtype: long
            The indices of (relation, object) pairs.

        :return: torch.Tensor, shape: (batch_size, num_entities), dtype: float
            For each p-o pair, the scores for all possible subjects.
        """
        raise NotImplementedError

    @abstractmethod
    def init_empty_weights_(self) -> 'BaseModule':
        """Initialize all uninitialized weights and embeddings."""
        raise NotImplementedError

    @abstractmethod
    def clear_weights_(self) -> 'BaseModule':
        """Clear all weights and embeddings."""
        raise NotImplementedError

    def reset_weights_(self) -> 'BaseModule':
        """Force re-initialization of all weights."""
        return self.clear_weights_().init_empty_weights_().to_device_()

    def get_grad_params(self) -> Iterable[nn.Parameter]:
        """Get the parameters that require gradients."""
        # TODO: Why do we need that? The optimizer takes care of filtering the parameters.
        return filter(lambda p: p.requires_grad, self.parameters())

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
            dimensions=self.embedding_dim,
        )

        embeddings = self.entity_embeddings.weight.detach().cpu().numpy()
        names = sorted(
            self.triples_factory.entity_to_id,
            key=self.triples_factory.entity_to_id.get
        )

        if use_tqdm:
            names = tqdm(names, desc='Building SQLAlchemy models')
        for name, embedding in zip(names, embeddings):
            embedding = Embedding(
                collection=collection,
                curie=name,
                vector=list(embedding),
            )
            session.add(embedding)
        session.add(collection)
        session.commit()
        return collection


class RegularizedModel(BaseModule):
    """Adds regularization to the BaseModule.

    The class modifies the loss method by adding a weighted regularization term. The computation of the regularization
    term is left to the subclass that usually computes such in every forward pass.
    """

    #: Current regularization term
    #: Except for initialization to None, all management is left to the subclass.
    current_regularization_term: Optional[torch.FloatTensor]

    def __init__(
        self,
        *args,
        regularization_weight: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.regularization_weight = regularization_weight
        self.current_regularization_term = None

    def compute_label_loss(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor
    ) -> torch.FloatTensor:  # noqa: D102
        assert self.current_regularization_term is not None, \
            "Regularized models have to set 'self.current_regularization_term' in the forward pass"
        loss = super().compute_label_loss(predictions=predictions, labels=labels)
        loss += self.regularization_weight * self.current_regularization_term
        return loss

    def compute_mr_loss(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        assert self.current_regularization_term is not None, \
            "Regularized models have to set 'self.current_regularization_term' in the forward pass"
        loss = super().compute_mr_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        loss += self.regularization_weight * self.current_regularization_term
        return loss
