# -*- coding: utf-8 -*-

"""Base module for all KGE models."""

import functools
import inspect
import itertools as itt
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, ClassVar, Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..losses import Loss, MarginRankingLoss, NSSALoss
from ..nn import Embedding
from ..regularizers import NoRegularizer, Regularizer
from ..triples import TriplesFactory
from ..typing import Constrainer, DeviceHint, Initializer, MappedTriples, Normalizer
from ..utils import NoRandomSeedNecessary, resolve_device, set_random_seed

__all__ = [
    'Model',
    'EntityEmbeddingModel',
    'EntityRelationEmbeddingModel',
    'MultimodalModel',
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


def get_novelty_mask(
    mapped_triples: MappedTriples,
    query_ids: np.ndarray,
    col: int,
    other_col_ids: Tuple[int, int],
) -> np.ndarray:
    r"""Calculate for each query ID whether it is novel.

    In particular, computes:

    .. math ::
        q \notin \{t[col] in T \mid t[\neg col] = p\}

    for each q in query_ids where :math:`\neg col` denotes all columns but `col`, and `p` equals `other_col_ids`.

    :param mapped_triples: shape: (num_triples, 3), dtype: long
        The mapped triples (i.e. ID-based).
    :param query_ids: shape: (num_queries,), dtype: long
        The query IDs. Are assumed to be unique (i.e. without duplicates).
    :param col:
        The column to which the query IDs correspond.
    :param other_col_ids:
        Fixed IDs for the other columns.

    :return: shape: (num_queries,), dtype: bool
        A boolean mask indicating whether the ID does not correspond to a known triple.
    """
    other_cols = sorted(set(range(mapped_triples.shape[1])).difference({col}))
    other_col_ids = torch.tensor(data=other_col_ids, dtype=torch.long, device=mapped_triples.device)
    filter_mask = (mapped_triples[:, other_cols] == other_col_ids[None, :]).all(dim=-1)
    known_ids = mapped_triples[filter_mask, col].unique().cpu().numpy()
    return np.isin(element=query_ids, test_elements=known_ids, assume_unique=True, invert=True)


def get_novelty_all_mask(
    mapped_triples: MappedTriples,
    query: np.ndarray,
) -> np.ndarray:
    known = {tuple(triple) for triple in mapped_triples.tolist()}
    return np.asarray(
        [tuple(triple) not in known for triple in query],
        dtype=np.bool,
    )


def _postprocess_prediction_df(
    rv: pd.DataFrame,
    *,
    col: int,
    add_novelties: bool,
    remove_known: bool,
    training: Optional[torch.LongTensor],
    testing: Optional[torch.LongTensor],
    query_ids_key: str,
    other_col_ids: Tuple[int, int],
) -> pd.DataFrame:
    if add_novelties or remove_known:
        rv['in_training'] = ~get_novelty_mask(
            mapped_triples=training,
            query_ids=rv[query_ids_key],
            col=col,
            other_col_ids=other_col_ids,
        )
    if add_novelties and testing is not None:
        rv['in_testing'] = ~get_novelty_mask(
            mapped_triples=testing,
            query_ids=rv[query_ids_key],
            col=col,
            other_col_ids=other_col_ids,
        )
    return _process_remove_known(rv, remove_known, testing)


def _postprocess_prediction_all_df(
    df: pd.DataFrame,
    *,
    add_novelties: bool,
    remove_known: bool,
    training: Optional[torch.LongTensor],
    testing: Optional[torch.LongTensor],
) -> pd.DataFrame:
    if add_novelties or remove_known:
        assert training is not None
        df['in_training'] = ~get_novelty_all_mask(
            mapped_triples=training,
            query=df[['head_id', 'relation_id', 'tail_id']].values,
        )
    if add_novelties and testing is not None:
        assert testing is not None
        df['in_testing'] = ~get_novelty_all_mask(
            mapped_triples=testing,
            query=df[['head_id', 'relation_id', 'tail_id']].values,
        )
    return _process_remove_known(df, remove_known, testing)


def _process_remove_known(df: pd.DataFrame, remove_known: bool, testing: Optional[torch.LongTensor]) -> pd.DataFrame:
    if not remove_known:
        return df

    df = df[~df['in_training']]
    del df['in_training']
    if testing is None:
        return df

    df = df[~df['in_testing']]
    del df['in_testing']
    return df


def _track_hyperparameters(cls: Type['Model']) -> None:
    """Initialize the subclass while keeping track of hyper-parameters."""
    # Keep track of the hyper-parameters that are used across all
    # subclasses of BaseModule
    for k in cls.__init__.__annotations__.keys():
        if k not in Model.__init__.__annotations__:
            Model._hyperparameter_usage[k].add(cls.__name__)


def _add_post_reset_parameters(cls: Type['Model']) -> None:
    # The following lines add in a post-init hook to all subclasses
    # such that the reset_parameters_() function is run
    _original_init = cls.__init__

    @functools.wraps(_original_init)
    def _new_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        self.reset_parameters_()

    cls.__init__ = _new_init


class Model(nn.Module, ABC):
    """A base module for all of the KGE models."""

    #: A dictionary of hyper-parameters to the models that use them
    _hyperparameter_usage: ClassVar[Dict[str, Set[str]]] = defaultdict(set)

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]]
    #: The default loss function class
    loss_default: ClassVar[Type[Loss]] = MarginRankingLoss
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
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        automatic_memory_optimization: Optional[bool] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        """Initialize the module.

        :param triples_factory:
            The triples factory facilitates access to the dataset.
        :param loss:
            The loss to use. If None is given, use the loss default specific to the model subclass.
        :param predict_with_sigmoid:
            Whether to apply sigmoid onto the scores when predicting scores. Applying sigmoid at prediction time may
            lead to exactly equal scores for certain triples with very high, or very low score. When not trained with
            applying sigmoid (or using BCEWithLogitsLoss), the scores are not calibrated to perform well with sigmoid.
        :param automatic_memory_optimization:
            If set to `True`, the model derives the maximum possible batch sizes for the scoring of triples during
            evaluation and also training (if no batch size was given). This allows to fully utilize the hardware at hand
            and achieves the fastest calculations possible.
        :param preferred_device:
            The preferred device for model training and inference.
        :param random_seed:
            A random seed to use for initialising the model's weights. **Should** be set when aiming at reproducibility.
        :param regularizer:
            A regularizer to use for training.
        """
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
        self.is_mr_loss = isinstance(self.loss, MarginRankingLoss)

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

        '''
        When predict_with_sigmoid is set to True, the sigmoid function is applied to the logits during evaluation and
        also for predictions after training, but has no effect on the training.
        '''
        self.predict_with_sigmoid = predict_with_sigmoid

        # This allows to store the optimized parameters
        self.automatic_memory_optimization = automatic_memory_optimization

    @classmethod
    def _is_abstract(cls) -> bool:
        return inspect.isabstract(cls)

    def __init_subclass__(cls, reset_parameters_post_init: bool = True, **kwargs):  # noqa:D105
        if not cls._is_abstract():
            _track_hyperparameters(cls)
            if reset_parameters_post_init:
                _add_post_reset_parameters(cls)

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

    @abstractmethod
    def _reset_parameters_(self):  # noqa: D401
        """Reset all parameters of the model in-place."""
        raise NotImplementedError

    def reset_parameters_(self) -> 'Model':  # noqa: D401
        """Reset all parameters of the model and enforce model constraints."""
        self._reset_parameters_()
        self.to_device_()
        self.post_parameter_update()
        return self

    @property
    def num_entities(self) -> int:  # noqa: D401
        """The number of entities in the knowledge graph."""
        return self.triples_factory.num_entities

    @property
    def num_relations(self) -> int:  # noqa: D401
        """The number of unique relation types in the knowledge graph."""
        return self.triples_factory.num_relations

    def _set_device(self, device: DeviceHint = None) -> None:
        """Set the Torch device to use."""
        self.device = resolve_device(device=device)

    def to_device_(self) -> 'Model':
        """Transfer model to device."""
        self.to(self.device)
        self.regularizer.to(self.device)
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

        :param triples: shape: (number of triples, 3), dtype: long
            The indices of (head, relation, tail) triples.

        :return: shape: (number of triples, 1), dtype: float
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

        :param hr_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, relation) pairs.
        :param slice_size: >0
            The divisor for the scoring function when using slicing.

        :return: shape: (batch_size, num_entities), dtype: float
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

    def predict_heads(
        self,
        relation_label: str,
        tail_label: str,
        add_novelties: bool = True,
        remove_known: bool = False,
        testing: Optional[torch.LongTensor] = None,
    ) -> pd.DataFrame:
        """Predict tails for the given head and relation (given by label).

        :param relation_label: The string label for the relation
        :param tail_label: The string label for the tail entity
        :param add_novelties: Should the dataframe include a column denoting if the ranked head entities correspond
         to novel triples?
        :param remove_known: Should non-novel triples (those appearing in the training set) be shown with the results?
         On one hand, this allows you to better assess the goodness of the predictions - you want to see that the
         non-novel triples generally have higher scores. On the other hand, if you're doing hypothesis generation, they
         may pose as a distraction. If this is set to True, then non-novel triples will be removed and the column
         denoting novelty will be excluded, since all remaining triples will be novel. Defaults to false.
        :param testing: The mapped_triples from the testing triples factory (TriplesFactory.mapped_triples)

        The following example shows that after you train a model on the Nations dataset,
        you can score all entities w.r.t a given relation and tail entity.

        >>> from pykeen.pipeline import pipeline
        >>> result = pipeline(
        ...     dataset='Nations',
        ...     model='RotatE',
        ... )
        >>> df = result.model.predict_heads('accusation', 'brazil')
        """
        tail_id = self.triples_factory.entity_to_id[tail_label]
        relation_id = self.triples_factory.relation_to_id[relation_label]
        rt_batch = torch.tensor([[relation_id, tail_id]], dtype=torch.long, device=self.device)
        scores = self.predict_scores_all_heads(rt_batch)
        scores = scores[0, :].tolist()
        rv = pd.DataFrame(
            [
                (entity_id, entity_label, scores[entity_id])
                for entity_label, entity_id in self.triples_factory.entity_to_id.items()
            ],
            columns=['head_id', 'head_label', 'score'],
        ).sort_values('score', ascending=False)

        return _postprocess_prediction_df(
            rv=rv,
            add_novelties=add_novelties,
            remove_known=remove_known,
            training=self.triples_factory.mapped_triples,
            testing=testing,
            query_ids_key='head_id',
            col=0,
            other_col_ids=(relation_id, tail_id),
        )

    def predict_tails(
        self,
        head_label: str,
        relation_label: str,
        add_novelties: bool = True,
        remove_known: bool = False,
        testing: Optional[torch.LongTensor] = None,
    ) -> pd.DataFrame:
        """Predict tails for the given head and relation (given by label).

        :param head_label: The string label for the head entity
        :param relation_label: The string label for the relation
        :param add_novelties: Should the dataframe include a column denoting if the ranked tail entities correspond
         to novel triples?
        :param remove_known: Should non-novel triples (those appearing in the training set) be shown with the results?
         On one hand, this allows you to better assess the goodness of the predictions - you want to see that the
         non-novel triples generally have higher scores. On the other hand, if you're doing hypothesis generation, they
         may pose as a distraction. If this is set to True, then non-novel triples will be removed and the column
         denoting novelty will be excluded, since all remaining triples will be novel. Defaults to false.
        :param testing: The mapped_triples from the testing triples factory (TriplesFactory.mapped_triples)

        The following example shows that after you train a model on the Nations dataset,
        you can score all entities w.r.t a given head entity and relation.

        >>> from pykeen.pipeline import pipeline
        >>> result = pipeline(
        ...     dataset='Nations',
        ...     model='RotatE',
        ... )
        >>> df = result.model.predict_tails('brazil', 'accusation')
        """
        head_id = self.triples_factory.entity_to_id[head_label]
        relation_id = self.triples_factory.relation_to_id[relation_label]
        batch = torch.tensor([[head_id, relation_id]], dtype=torch.long, device=self.device)
        scores = self.predict_scores_all_tails(batch)
        scores = scores[0, :].tolist()
        rv = pd.DataFrame(
            [
                (entity_id, entity_label, scores[entity_id])
                for entity_label, entity_id in self.triples_factory.entity_to_id.items()
            ],
            columns=['tail_id', 'tail_label', 'score'],
        ).sort_values('score', ascending=False)

        return _postprocess_prediction_df(
            rv,
            add_novelties=add_novelties,
            remove_known=remove_known,
            testing=testing,
            training=self.triples_factory.mapped_triples,
            query_ids_key='tail_id',
            col=2,
            other_col_ids=(head_id, relation_id),
        )

    def predict_scores_all_relations(
        self,
        ht_batch: torch.LongTensor,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        """Forward pass using middle (relation) prediction for obtaining scores of all possible relations.

        This method calculates the score for all possible relations for each (head, tail) pair.

        Additionally, the model is set to evaluation mode.

        :param ht_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, tail) pairs.
        :param slice_size: >0
            The divisor for the scoring function when using slicing.

        :return: shape: (batch_size, num_relations), dtype: float
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

        :param rt_batch: shape: (batch_size, 2), dtype: long
            The indices of (relation, tail) pairs.
        :param slice_size: >0
            The divisor for the scoring function when using slicing.

        :return: shape: (batch_size, num_entities), dtype: float
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
        The PyKEEN package handles _inverse relations_ by adding the number of relations to the indices of the
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

    def _score_all_triples(
        self,
        batch_size: int = 1,
        return_tensors: bool = False,
        *,
        add_novelties: bool = True,
        remove_known: bool = False,
        testing: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.LongTensor, torch.FloatTensor], pd.DataFrame]:
        """Compute and store scores for all triples.

        :return: Parallel arrays of triples and scores
        """
        # initialize buffer on cpu
        scores = torch.empty(self.num_relations, self.num_entities, self.num_entities, dtype=torch.float32)
        assert self.num_entities ** 2 * self.num_relations < (2 ** 63 - 1)

        for r, e in itt.product(
            range(self.num_relations),
            range(0, self.num_entities, batch_size),
        ):
            # calculate batch scores
            hs = torch.arange(e, min(e + batch_size, self.num_entities), device=self.device)
            hr_batch = torch.stack([
                hs,
                hs.new_empty(1).fill_(value=r).repeat(hs.shape[0]),
            ], dim=-1)
            scores[r, e:e + batch_size, :] = self.predict_scores_all_tails(hr_batch=hr_batch).to(scores.device)

        # Explicitly create triples
        triples = torch.stack([
            torch.arange(self.num_relations).view(-1, 1, 1).repeat(1, self.num_entities, self.num_entities),
            torch.arange(self.num_entities).view(1, -1, 1).repeat(self.num_relations, 1, self.num_entities),
            torch.arange(self.num_entities).view(1, 1, -1).repeat(self.num_relations, self.num_entities, 1),
        ], dim=-1).view(-1, 3)[:, [1, 0, 2]]

        # Sort final result
        scores, ind = torch.sort(scores.flatten(), descending=True)
        triples = triples[ind]

        if return_tensors:
            return triples, scores

        rv = self.make_labeled_df(triples, score=scores)
        return _postprocess_prediction_all_df(
            df=rv,
            add_novelties=add_novelties,
            remove_known=remove_known,
            training=self.triples_factory.mapped_triples,
            testing=testing,
        )

    def score_all_triples(
        self,
        k: Optional[int] = None,
        batch_size: int = 1,
        return_tensors: bool = False,
        add_novelties: bool = True,
        remove_known: bool = False,
        testing: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.LongTensor, torch.FloatTensor], pd.DataFrame]:
        """Compute scores for all triples, optionally returning only the k highest scoring.

        .. note:: This operation is computationally very expensive for reasonably-sized knowledge graphs.
        .. warning:: Setting k=None may lead to huge memory requirements.

        :param k:
            The number of triples to return. Set to None, to keep all.

        :param batch_size:
            The batch size to use for calculating scores.

        :return: shape: (k, 3)
            A tensor containing the k highest scoring triples, or all possible triples if k=None.

        Example usage:

        .. code-block:: python

            from pykeen.pipeline import pipeline

            # Train a model (quickly)
            result = pipeline(model='RotatE', dataset='Nations', training_kwargs=dict(num_epochs=5))
            model = result.model

            # Get scores for *all* triples
            tensor = model.score_all_triples()
            df = model.make_labeled_df(tensor)

            # Get scores for top 15 triples
            top_df = model.score_all_triples(k=15)
        """
        # set model to evaluation mode
        self.eval()

        # Do not track gradients
        with torch.no_grad():
            logger.warning(
                f'score_all_triples is an expensive operation, involving {self.num_entities ** 2 * self.num_relations} '
                f'score evaluations.',
            )

            if k is None:
                logger.warning(
                    'Not providing k to score_all_triples entails huge memory requirements for reasonably-sized '
                    'knowledge graphs.',
                )
                return self._score_all_triples(
                    batch_size=batch_size,
                    return_tensors=return_tensors,
                    testing=testing,
                    add_novelties=add_novelties,
                    remove_known=remove_known,
                )

            # initialize buffer on device
            result = torch.ones(0, 3, dtype=torch.long, device=self.device)
            scores = torch.empty(0, dtype=torch.float32, device=self.device)

            for r, e in itt.product(
                range(self.num_relations),
                range(0, self.num_entities, batch_size),
            ):
                # calculate batch scores
                hs = torch.arange(e, min(e + batch_size, self.num_entities), device=self.device)
                real_batch_size = hs.shape[0]
                hr_batch = torch.stack([
                    hs,
                    hs.new_empty(1).fill_(value=r).repeat(real_batch_size),
                ], dim=-1)
                top_scores = self.predict_scores_all_tails(hr_batch=hr_batch).view(-1)

                # get top scores within batch
                if top_scores.numel() >= k:
                    top_scores, top_indices = top_scores.topk(k=min(k, batch_size), largest=True, sorted=False)
                    top_heads, top_tails = top_indices // self.num_entities, top_indices % self.num_entities
                else:
                    top_heads = hs.view(-1, 1).repeat(1, self.num_entities).view(-1)
                    top_tails = torch.arange(self.num_entities, device=hs.device).view(1, -1).repeat(
                        real_batch_size, 1).view(-1)

                top_triples = torch.stack([
                    top_heads,
                    top_heads.new_empty(top_heads.shape).fill_(value=r),
                    top_tails,
                ], dim=-1)

                # append to global top scores
                scores = torch.cat([scores, top_scores])
                result = torch.cat([result, top_triples])

                # reduce size if necessary
                if result.shape[0] > k:
                    scores, indices = scores.topk(k=k, largest=True, sorted=False)
                    result = result[indices]

            # Sort final result
            scores, indices = torch.sort(scores, descending=True)
            result = result[indices]

        if return_tensors:
            return result, scores

        rv = self.make_labeled_df(result, score=scores)
        return _postprocess_prediction_all_df(
            df=rv,
            add_novelties=add_novelties,
            remove_known=remove_known,
            training=self.triples_factory.mapped_triples,
            testing=testing,
        )

    def make_labeled_df(
        self,
        tensor: torch.LongTensor,
        **kwargs: Union[torch.Tensor, np.ndarray, Sequence],
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
        return self.triples_factory.tensor_to_df(tensor, **kwargs)

    def post_parameter_update(self) -> None:
        """Has to be called after each parameter update."""
        self.regularizer.reset()

    def regularize_if_necessary(self, *tensors: torch.FloatTensor) -> None:
        """Update the regularizer's term given some tensors, if regularization is requested.

        :param tensors: The tensors that should be passed to the regularizer to update its term.
        """
        if self.training:
            self.regularizer.update(*tensors)

    def compute_mr_loss(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute the mean ranking loss for the positive and negative scores.

        :param positive_scores:  shape: s, dtype: float
            The scores for positive triples.
        :param negative_scores: shape: s, dtype: float
            The scores for negative triples.
        :raises RuntimeError:
            If the chosen loss function does not allow the calculation of margin ranking
        :return: dtype: float, scalar
            The margin ranking loss value.
        """
        if not self.is_mr_loss:
            raise RuntimeError(
                'The chosen loss does not allow the calculation of margin ranking'
                ' losses. Please use the compute_loss method instead.',
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

        :return: dtype: float, scalar
            The label loss value.
        """
        return self._compute_loss(tensor_1=predictions, tensor_2=labels)

    def compute_self_adversarial_negative_sampling_loss(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute self adversarial negative sampling loss.

        :param positive_scores: shape: s
            The tensor containing the positive scores.
        :param negative_scores: shape: s
            Tensor containing the negative scores.
        :raises RuntimeError:
            If the chosen loss does not allow the calculation of self adversarial negative sampling losses.
        :return: dtype: float, scalar
            The loss value.
        """
        if not self.is_nssa_loss:
            raise RuntimeError(
                'The chosen loss does not allow the calculation of self adversarial negative sampling'
                ' losses. Please use the compute_self_adversarial_negative_sampling_loss method instead.',
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
        :raises RuntimeError:
            If the chosen loss does not allow the calculation of margin label losses.
        :return: dtype: float, scalar
            The label loss value.
        """
        if self.is_mr_loss:
            raise RuntimeError(
                'The chosen loss does not allow the calculation of margin label'
                ' losses. Please use the compute_mr_loss method instead.',
            )
        return self.loss(tensor_1, tensor_2) + self.regularizer.term

    @abstractmethod
    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass.

        This method takes head, relation and tail of each triple and calculates the corresponding score.

        :param hrt_batch: shape: (batch_size, 3), dtype: long
            The indices of (head, relation, tail) triples.
        :raises NotImplementedError:
            If the method was not implemented for this class.
        :return: shape: (batch_size, 1), dtype: float
            The score for each triple.
        """
        raise NotImplementedError

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass using right side (tail) prediction.

        This method calculates the score for all possible tails for each (head, relation) pair.

        :param hr_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, relation) pairs.

        :return: shape: (batch_size, num_entities), dtype: float
            For each h-r pair, the scores for all possible tails.
        """
        logger.warning(
            'Calculations will fall back to using the score_hrt method, since this model does not have a specific '
            'score_t function. This might cause the calculations to take longer than necessary.',
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

        :param rt_batch: shape: (batch_size, 2), dtype: long
            The indices of (relation, tail) pairs.

        :return: shape: (batch_size, num_entities), dtype: float
            For each r-t pair, the scores for all possible heads.
        """
        logger.warning(
            'Calculations will fall back to using the score_hrt method, since this model does not have a specific '
            'score_h function. This might cause the calculations to take longer than necessary.',
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

        :param ht_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, tail) pairs.

        :return: shape: (batch_size, num_relations), dtype: float
            For each h-t pair, the scores for all possible relations.
        """
        logger.warning(
            'Calculations will fall back to using the score_hrt method, since this model does not have a specific '
            'score_r function. This might cause the calculations to take longer than necessary.',
        )
        # Extend the ht_batch such that each (h, t) pair is combined with all possible relations
        hrt_batch = _extend_batch(batch=ht_batch, all_ids=list(self.triples_factory.relation_to_id.values()), dim=1)
        # Calculate the scores for each (h, r, t) triple using the generic interaction function
        expanded_scores = self.score_hrt(hrt_batch=hrt_batch)
        # Reshape the scores to match the pre-defined output shape of the score_r function.
        scores = expanded_scores.view(ht_batch.shape[0], -1)
        return scores

    def get_grad_params(self) -> Iterable[nn.Parameter]:
        """Get the parameters that require gradients."""
        # TODO: Why do we need that? The optimizer takes care of filtering the parameters.
        return filter(lambda p: p.requires_grad, self.parameters())

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


class EntityEmbeddingModel(Model):
    """A base module for most KGE models that have one embedding for entities."""

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        automatic_memory_optimization: Optional[bool] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
        entity_initializer: Optional[Initializer] = None,
        entity_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        entity_normalizer: Optional[Normalizer] = None,
        entity_normalizer_kwargs: Optional[Mapping[str, Any]] = None,
        entity_constrainer: Optional[Constrainer] = None,
        entity_constrainer_kwargs: Optional[Mapping[str, Any]] = None,

    ) -> None:
        """Initialize the entity embedding model.

        :param embedding_dim:
            The embedding dimensionality. Exact usages depends on the specific model subclass.

        .. seealso:: Constructor of the base class :class:`pykeen.models.Model`
        """
        super().__init__(
            triples_factory=triples_factory,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            predict_with_sigmoid=predict_with_sigmoid,
        )
        self.entity_embeddings = Embedding.init_with_device(
            num_embeddings=triples_factory.num_entities,
            embedding_dim=embedding_dim,
            device=self.device,
            initializer=entity_initializer,
            initializer_kwargs=entity_initializer_kwargs,
            normalizer=entity_normalizer,
            normalizer_kwargs=entity_normalizer_kwargs,
            constrainer=entity_constrainer,
            constrainer_kwargs=entity_constrainer_kwargs,
        )

    @property
    def embedding_dim(self) -> int:  # noqa:D401
        """The entity embedding dimension."""
        return self.entity_embeddings.embedding_dim

    def _reset_parameters_(self):  # noqa: D102
        self.entity_embeddings.reset_parameters()

    def post_parameter_update(self) -> None:  # noqa: D102
        # make sure to call this first, to reset regularizer state!
        super().post_parameter_update()
        self.entity_embeddings.post_parameter_update()


class EntityRelationEmbeddingModel(Model):
    """A base module for KGE models that have different embeddings for entities and relations."""

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        relation_dim: Optional[int] = None,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        automatic_memory_optimization: Optional[bool] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
        entity_initializer: Optional[Initializer] = None,
        entity_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        entity_normalizer: Optional[Normalizer] = None,
        entity_normalizer_kwargs: Optional[Mapping[str, Any]] = None,
        entity_constrainer: Optional[Constrainer] = None,
        entity_constrainer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Optional[Initializer] = None,
        relation_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_normalizer: Optional[Normalizer] = None,
        relation_normalizer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_constrainer: Optional[Constrainer] = None,
        relation_constrainer_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Initialize the entity embedding model.

        :param relation_dim:
            The relation embedding dimensionality. If not given, defaults to same size as entity embedding
            dimension.

        .. seealso:: Constructor of the base class :class:`pykeen.models.Model`
        .. seealso:: Constructor of the base class :class:`pykeen.models.EntityEmbeddingModel`
        """
        super().__init__(
            triples_factory=triples_factory,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            predict_with_sigmoid=predict_with_sigmoid,
        )
        self.entity_embeddings = Embedding.init_with_device(
            num_embeddings=triples_factory.num_entities,
            embedding_dim=embedding_dim,
            device=self.device,
            initializer=entity_initializer,
            initializer_kwargs=entity_initializer_kwargs,
            normalizer=entity_normalizer,
            normalizer_kwargs=entity_normalizer_kwargs,
            constrainer=entity_constrainer,
            constrainer_kwargs=entity_constrainer_kwargs,
        )

        # Default for relation dimensionality
        if relation_dim is None:
            relation_dim = embedding_dim

        self.relation_embeddings = Embedding.init_with_device(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=relation_dim,
            device=self.device,
            initializer=relation_initializer,
            initializer_kwargs=relation_initializer_kwargs,
            normalizer=relation_normalizer,
            normalizer_kwargs=relation_normalizer_kwargs,
            constrainer=relation_constrainer,
            constrainer_kwargs=relation_constrainer_kwargs,
        )

    @property
    def embedding_dim(self) -> int:  # noqa:D401
        """The entity embedding dimension."""
        return self.entity_embeddings.embedding_dim

    @property
    def relation_dim(self):  # noqa:D401
        """The relation embedding dimension."""
        return self.relation_embeddings.embedding_dim

    def _reset_parameters_(self):  # noqa: D102
        self.entity_embeddings.reset_parameters()
        self.relation_embeddings.reset_parameters()

    def post_parameter_update(self) -> None:  # noqa: D102
        # make sure to call this first, to reset regularizer state!
        super().post_parameter_update()
        self.entity_embeddings.post_parameter_update()
        self.relation_embeddings.post_parameter_update()


def _can_slice(fn) -> bool:
    return 'slice_size' in inspect.getfullargspec(fn).args


class MultimodalModel(EntityRelationEmbeddingModel):
    """A multimodal KGE model."""
