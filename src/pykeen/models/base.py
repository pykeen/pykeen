# -*- coding: utf-8 -*-

"""Base module for all KGE models."""

import inspect
import logging
from abc import abstractmethod
from collections import defaultdict
from typing import Any, ClassVar, Collection, Dict, Iterable, List, Mapping, Optional, Set, Type, Union

import torch
from torch import nn
from tqdm import tqdm

from ..losses import Loss, NSSALoss
from ..regularizers import NoRegularizer, Regularizer
from ..triples import TriplesFactory
from ..typing import MappedTriples
from ..utils import NoRandomSeedNecessary, resolve_device, set_random_seed
from ..version import get_version

__all__ = [
    'Model',
]

logger = logging.getLogger(__name__)

UNSUPPORTED_FOR_SUBBATCHING = (  # must be a tuple
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


def _extend_batch(
    batch: MappedTriples,
    all_ids: List[int],
    dim: int,
) -> MappedTriples:
    """Extend batch for 1-to-all scoring by explicit enumeration.

    :param batch: shape: (batch_size, 2)
        The batch.
    :param all_ids: len: num_choices
        The IDs to enumerate.
    :param dim: in {0,1,2}
        The column along which to insert the enumerated IDs.

    :return: shape: (batch_size * num_choices, 3)
        A large batch, where every pair from the original batch is combined with every ID.
    """
    # Extend the batch to the number of IDs such that each pair can be combined with all possible IDs
    extended_batch = batch.repeat_interleave(repeats=len(all_ids), dim=0)

    # Create a tensor of all IDs
    ids = torch.tensor(all_ids, dtype=torch.long, device=batch.device)

    # Extend all IDs to the number of pairs such that each ID can be combined with every pair
    extended_ids = ids.repeat(batch.shape[0])

    # Fuse the extended pairs with all IDs to a new (h, r, t) triple tensor.
    columns = [extended_batch[:, i] for i in (0, 1)]
    columns.insert(dim, extended_ids)
    hrt_batch = torch.stack(columns, dim=-1)

    return hrt_batch


class Model(nn.Module):
    """A base module for all of the KGE models."""

    #: A dictionary of hyper-parameters to the models that use them
    _hyperparameter_usage: ClassVar[Dict[str, Set[str]]] = defaultdict(set)

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]]
    #: The default loss function class
    loss_default: ClassVar[Type[Loss]] = nn.MarginRankingLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Optional[Mapping[str, Any]]] = dict(margin=1.0, reduction='mean')
    #: The instance of the loss
    loss: Loss
    #: The default regularizer class
    regularizer_default: ClassVar[Type[Regularizer]] = NoRegularizer
    #: The default parameters for the default regularizer class
    regularizer_default_kwargs: ClassVar[Optional[Mapping[str, Any]]] = None
    #: The instance of the regularizer
    regularizer: Regularizer

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        entity_embeddings: Optional[nn.Embedding] = None,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        automatic_memory_optimization: Optional[bool] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        """Initialize the module."""
        super().__init__()

        # Initialize the device
        self._set_device(preferred_device)

        # Random seeds have to set before the embeddings are initialized
        if random_seed is None:
            logger.warning('No random seed is specified. This may lead to non-reproducible results.')
        elif random_seed is not NoRandomSeedNecessary:
            set_random_seed(random_seed)

        if automatic_memory_optimization is None:
            automatic_memory_optimization = True

        # Loss
        if loss is None:
            self.loss = self.loss_default(**self.loss_default_kwargs)
        else:
            self.loss = loss

        # TODO: Check loss functions that require 1 and -1 as label but only
        self.is_mr_loss = isinstance(self.loss, nn.MarginRankingLoss)

        # Regularizer
        if regularizer is None:
            regularizer = self.regularizer_default(
                device=self.device,
                **(self.regularizer_default_kwargs or {}),
            )
        self.regularizer = regularizer

        self.is_nssa_loss = isinstance(self.loss, NSSALoss)

        # The triples factory facilitates access to the dataset.
        self.triples_factory = triples_factory

        #: The dimension of the embeddings to generate
        self.embedding_dim = embedding_dim

        # The embeddings are first initiated when calling the fit function
        self.entity_embeddings = entity_embeddings

        '''
        When predict_with_sigmoid is set to True, the sigmoid function is applied to the logits during evaluation and
        also for predictions after training, but has no effect on the training.
        '''
        self.predict_with_sigmoid = predict_with_sigmoid

        # This allows to store the optimized parameters
        self.automatic_memory_optimization = automatic_memory_optimization

    @property
    def can_slice_h(self) -> bool:
        """Whether score_h supports slicing."""
        return _can_slice(self.score_h)

    @property
    def can_slice_r(self) -> bool:
        """Whether score_r supports slicing."""
        return _can_slice(self.score_r)

    @property
    def can_slice_t(self) -> bool:
        """Whether score_t supports slicing."""
        return _can_slice(self.score_t)

    @property
    def modules_not_supporting_sub_batching(self) -> Collection[nn.Module]:
        """Return all modules not supporting sub-batching."""
        return [
            module
            for module in self.modules()
            if isinstance(module, UNSUPPORTED_FOR_SUBBATCHING)
        ]

    @property
    def supports_subbatching(self) -> bool:  # noqa: D400, D401
        """Does this model support sub-batching?"""
        return len(self.modules_not_supporting_sub_batching) == 0

    def _init_weights_on_device(self):  # noqa: D401
        """A hook called after initialization."""
        self.init_empty_weights_()
        self.to_device_()
        self.post_parameter_update()

    def __init_subclass__(cls, **kwargs):
        """Initialize the subclass while keeping track of hyper-parameters."""
        super().__init_subclass__(**kwargs)

        # Keep track of the hyper-parameters that are used across all
        # subclasses of BaseModule
        for k in cls.__init__.__annotations__.keys():
            if k not in Model.__init__.__annotations__:
                Model._hyperparameter_usage[k].add(cls.__name__)

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

    def to_device_(self) -> 'Model':
        """Transfer model to device."""
        self.to(self.device)
        torch.cuda.empty_cache()
        return self

    def to_cpu_(self) -> 'Model':
        """Transfer the entire model to CPU."""
        self._set_device('cpu')
        return self.to_device_()

    def to_gpu_(self) -> 'Model':
        """Transfer the entire model to GPU."""
        self._set_device('cuda')
        return self.to_device_()

    def predict_scores(self, triples: torch.LongTensor) -> torch.FloatTensor:
        """Calculate the scores for triples.

        This method takes head, relation and tail of each triple and calculates the corresponding score.

        Additionally, the model is set to evaluation mode.

        :param triples: torch.Tensor, shape: (number of triples, 3), dtype: long
            The indices of (head, relation, tail) triples.

        :return: torch.Tensor, shape: (number of triples, 1), dtype: float
            The score for each triple.
        """
        # Enforce evaluation mode
        self.eval()
        scores = self.score_hrt(triples)
        if self.predict_with_sigmoid:
            scores = torch.sigmoid(scores)
        return scores

    def predict_scores_all_tails(
        self,
        hr_batch: torch.LongTensor,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        """Forward pass using right side (tail) prediction for obtaining scores of all possible tails.

        This method calculates the score for all possible tails for each (head, relation) pair.

        Additionally, the model is set to evaluation mode.

        :param hr_batch: torch.Tensor, shape: (batch_size, 2), dtype: long
            The indices of (head, relation) pairs.
        :param slice_size: >0
            The divisor for the scoring function when using slicing.

        :return: torch.Tensor, shape: (batch_size, num_entities), dtype: float
            For each h-r pair, the scores for all possible tails.
        """
        # Enforce evaluation mode
        self.eval()
        if slice_size is None:
            scores = self.score_t(hr_batch)
        else:
            scores = self.score_t(hr_batch, slice_size=slice_size)
        if self.predict_with_sigmoid:
            scores = torch.sigmoid(scores)
        return scores

    def predict_scores_all_relations(
        self,
        ht_batch: torch.LongTensor,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        """Forward pass using middle (relation) prediction for obtaining scores of all possible relations.

        This method calculates the score for all possible relations for each (head, tail) pair.

        Additionally, the model is set to evaluation mode.

        :param ht_batch: torch.Tensor, shape: (batch_size, 2), dtype: long
            The indices of (head, tail) pairs.
        :param slice_size: >0
            The divisor for the scoring function when using slicing.

        :return: torch.Tensor, shape: (batch_size, num_relations), dtype: float
            For each h-t pair, the scores for all possible relations.
        """
        # Enforce evaluation mode
        self.eval()
        if slice_size is None:
            scores = self.score_r(ht_batch)
        else:
            scores = self.score_r(ht_batch, slice_size=slice_size)
        if self.predict_with_sigmoid:
            scores = torch.sigmoid(scores)
        return scores

    def predict_scores_all_heads(
        self,
        rt_batch: torch.LongTensor,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        """Forward pass using left side (head) prediction for obtaining scores of all possible heads.

        This method calculates the score for all possible heads for each (relation, tail) pair.

        Additionally, the model is set to evaluation mode.

        :param rt_batch: torch.Tensor, shape: (batch_size, 2), dtype: long
            The indices of (relation, tail) pairs.
        :param slice_size: >0
            The divisor for the scoring function when using slicing.

        :return: torch.Tensor, shape: (batch_size, num_entities), dtype: float
            For each r-t pair, the scores for all possible heads.
        """
        # Enforce evaluation mode
        self.eval()

        '''
        In case the model was trained using inverse triples, the scoring of all heads is not handled by calculating
        the scores for all heads based on a (relation, tail) pair, but instead all possible tails are calculated
        for a (tail, inverse_relation) pair.
        '''
        if not self.triples_factory.create_inverse_triples:
            if slice_size is None:
                scores = self.score_h(rt_batch)
            else:
                scores = self.score_h(rt_batch, slice_size=slice_size)
            if self.predict_with_sigmoid:
                scores = torch.sigmoid(scores)
            return scores

        '''
        The PyKEEN package handles _inverse relations_ by adding the number of relations to the index of the
        _native relation_.
        Example:
        The triples/knowledge graph used to train the model contained 100 relations. Due to using inverse relations,
        the model now has an additional 100 inverse relations. If the _native relation_ has the index 3, the index
        of the _inverse relation_ is 4 (id of relation + 1).
        '''
        rt_batch_cloned = rt_batch.clone()
        rt_batch_cloned.to(device=rt_batch.device)

        # The number of relations stored in the triples factory includes the number of inverse relations
        # Id of inverse relation: relation + 1
        rt_batch_cloned[:, 0] = rt_batch_cloned[:, 0] + 1

        # The score_t function requires (entity, relation) pairs instead of (relation, entity) pairs
        rt_batch_cloned = rt_batch_cloned.flip(1)
        if slice_size is None:
            scores = self.score_t(rt_batch_cloned)
        else:
            scores = self.score_t(rt_batch_cloned, slice_size=slice_size)
        if self.predict_with_sigmoid:
            scores = torch.sigmoid(scores)
        return scores

    def post_parameter_update(self) -> None:
        """Has to be called after each parameter update."""
        self.regularizer.reset()

    def regularize_if_necessary(self, *tensors: torch.FloatTensor) -> None:
        """Update the regularizer's term given some tensors, if regularization is requested."""
        if self.training:
            self.regularizer.update(*tensors)

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
        if not self.is_mr_loss:
            raise RuntimeError(
                'The chosen loss does not allow the calculation of margin ranking'
                ' losses. Please use the compute_loss method instead.'
            )
        y = torch.ones_like(negative_scores, device=self.device)
        return self.loss(positive_scores, negative_scores, y) + self.regularizer.term

    def compute_label_loss(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute the classification loss.

        :param predictions: shape: s
            The tensor containing predictions.
        :param labels: shape: s
            The tensor containing labels.

        :return: torch.Tensor, dtype: float, scalar
            The label loss value.
        """
        return self._compute_loss(tensor_1=predictions, tensor_2=labels)

    def compute_self_adversarial_negative_sampling_loss(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Compute self adversarial negative sampling loss.

        :param positive_scores: shape: s
            The tensor containing the positive scores.
        :param negative_scores: shape: s
            Tensor containing the negative scores.
        :return: torch.Tensor, dtype: float, scalar
            The loss value.
        """
        if not self.is_nssa_loss:
            raise RuntimeError(
                'The chosen loss does not allow the calculation of self adversarial negative sampling'
                ' losses. Please use the compute_self_adversarial_negative_sampling_loss method instead.'
            )
        return self._compute_loss(tensor_1=positive_scores, tensor_2=negative_scores)

    def _compute_loss(
        self,
        tensor_1: torch.FloatTensor,
        tensor_2: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute the loss for functions requiring two separate tensors as input.

        :param tensor_1: shape: s
            The tensor containing predictions or positive scores.
        :param tensor_2: shape: s
            The tensor containing target values or the negative scores.

        :return: torch.Tensor, dtype: float, scalar
            The label loss value.
        """
        if self.is_mr_loss:
            raise RuntimeError(
                'The chosen loss does not allow the calculation of margin label'
                ' losses. Please use the compute_mr_loss method instead.'
            )
        return self.loss(tensor_1, tensor_2) + self.regularizer.term

    @abstractmethod
    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass.

        This method takes head, relation and tail of each triple and calculates the corresponding score.

        :param hrt_batch: torch.Tensor, shape: (batch_size, 3), dtype: long
            The indices of (head, relation, tail) triples.

        :return: torch.Tensor, shape: (batch_size, 1), dtype: float
            The score for each triple.
        """
        raise NotImplementedError

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass using right side (tail) prediction.

        This method calculates the score for all possible tails for each (head, relation) pair.

        :param hr_batch: torch.Tensor, shape: (batch_size, 2), dtype: long
            The indices of (head, relation) pairs.

        :return: torch.Tensor, shape: (batch_size, num_entities), dtype: float
            For each h-r pair, the scores for all possible tails.
        """
        logger.warning(
            'Calculations will fall back to using the score_hrt method, since this model does not have a specific '
            'score_t function. This might cause the calculations to take longer than necessary.'
        )
        # Extend the hr_batch such that each (h, r) pair is combined with all possible tails
        hrt_batch = _extend_batch(batch=hr_batch, all_ids=list(self.triples_factory.entity_to_id.values()), dim=2)
        # Calculate the scores for each (h, r, t) triple using the generic interaction function
        expanded_scores = self.score_hrt(hrt_batch=hrt_batch)
        # Reshape the scores to match the pre-defined output shape of the score_t function.
        scores = expanded_scores.view(hr_batch.shape[0], -1)
        return scores

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass using left side (head) prediction.

        This method calculates the score for all possible heads for each (relation, tail) pair.

        :param rt_batch: torch.Tensor, shape: (batch_size, 2), dtype: long
            The indices of (relation, tail) pairs.

        :return: torch.Tensor, shape: (batch_size, num_entities), dtype: float
            For each r-t pair, the scores for all possible heads.
        """
        logger.warning(
            'Calculations will fall back to using the score_hrt method, since this model does not have a specific '
            'score_h function. This might cause the calculations to take longer than necessary.'
        )
        # Extend the rt_batch such that each (r, t) pair is combined with all possible heads
        hrt_batch = _extend_batch(batch=rt_batch, all_ids=list(self.triples_factory.entity_to_id.values()), dim=0)
        # Calculate the scores for each (h, r, t) triple using the generic interaction function
        expanded_scores = self.score_hrt(hrt_batch=hrt_batch)
        # Reshape the scores to match the pre-defined output shape of the score_h function.
        scores = expanded_scores.view(rt_batch.shape[0], -1)
        return scores

    def score_r(self, ht_batch: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass using middle (relation) prediction.

        This method calculates the score for all possible relations for each (head, tail) pair.

        :param ht_batch: torch.Tensor, shape: (batch_size, 2), dtype: long
            The indices of (head, tail) pairs.

        :return: torch.Tensor, shape: (batch_size, num_relations), dtype: float
            For each h-t pair, the scores for all possible relations.
        """
        logger.warning(
            'Calculations will fall back to using the score_hrt method, since this model does not have a specific '
            'score_r function. This might cause the calculations to take longer than necessary.'
        )
        # Extend the ht_batch such that each (h, t) pair is combined with all possible relations
        hrt_batch = _extend_batch(batch=ht_batch, all_ids=list(self.triples_factory.relation_to_id.values()), dim=1)
        # Calculate the scores for each (h, r, t) triple using the generic interaction function
        expanded_scores = self.score_hrt(hrt_batch=hrt_batch)
        # Reshape the scores to match the pre-defined output shape of the score_r function.
        scores = expanded_scores.view(ht_batch.shape[0], -1)
        return scores

    @abstractmethod
    def init_empty_weights_(self) -> 'Model':
        """Initialize all uninitialized weights and embeddings."""
        raise NotImplementedError

    @abstractmethod
    def clear_weights_(self) -> 'Model':
        """Clear all weights and embeddings."""
        raise NotImplementedError

    def reset_weights_(self) -> 'Model':
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
            package_name='pykeen',
            package_version=get_version(),
            dimensions=self.embedding_dim,
        )

        embeddings = self.entity_embeddings.weight.detach().cpu().numpy()
        names = sorted(
            self.triples_factory.entity_to_id,
            key=self.triples_factory.entity_to_id.get,
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

    @property
    def num_parameter_bytes(self) -> int:
        """Calculate the number of bytes used for all parameters of the model."""
        return sum(p.numel() * p.element_size() for p in self.parameters(recurse=True))

    def save_state(self, path: str) -> None:
        """Save the state of the model.

        :param path:
            Path of the file where to store the state in.
        """
        torch.save(self.state_dict(), path)

    def load_state(self, path: str) -> None:
        """Load the state of the model.

        :param path:
            Path of the file where to load the state from.
        """
        self.load_state_dict(torch.load(path, map_location=self.device))


def _can_slice(fn) -> bool:
    return 'slice_size' in inspect.getfullargspec(fn).args
