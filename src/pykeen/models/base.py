# -*- coding: utf-8 -*-

"""Base module for all KGE models."""

import functools
import itertools as itt
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from operator import itemgetter
from typing import (
    Any, ClassVar, Collection, Dict, Generic, Iterable, List, Mapping, Optional, Sequence, Set, TYPE_CHECKING, Tuple,
    Type, Union,
)

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..losses import Loss, MarginRankingLoss, NSSALoss
from ..nn import Embedding, EmbeddingSpecification, RepresentationModule
from ..nn.modules import Interaction
from ..regularizers import Regularizer, collect_regularization_terms
from ..triples import TriplesFactory
from ..typing import DeviceHint, HeadRepresentation, MappedTriples, RelationRepresentation, TailRepresentation
from ..utils import NoRandomSeedNecessary, resolve_device, set_random_seed

if TYPE_CHECKING:
    from ..typing import Representation  # noqa

__all__ = [
    'Model',
    'ERModel',
    'SingleVectorEmbeddingModel',
    'DoubleRelationEmbeddingModel',
    'TwoVectorEmbeddingModel',
    'TwoSideEmbeddingModel',
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
    filter_mask = (mapped_triples[:, other_cols] == other_col_ids[None, :]).all(dim=-1)  # type: ignore
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

    # sorry mypy, but this kind of evil must be permitted.
    cls.__init__ = _new_init  # type: ignore


class Model(nn.Module, ABC):
    """An abstract class for knowledge graph embedding models (KGEMs).

    The only function that needs to be implemented for a given subclass is
    :meth:`Model.forward`. The job of the :meth:`Model.forward` function, as
    opposed to the completely general :meth:`torch.nn.Module.forward` is
    to take indices for the head, relation, and tails' respective representation(s)
    and to determine a score.

    Subclasses of Model can decide however they want on how to store entities' and
    relations' representations, how they want to be looked up, and how they should
    be scored. The :class:`ERModel` provides a commonly useful implementation
    which allows for the specification of one or more entity representations and
    one or more relation representations in the form of :class:`pykeen.nn.Embedding`
    as well as a matching instance of a :class:`pykeen.nn.Interaction`.
    """

    #: A dictionary of hyper-parameters to the models that use them
    _hyperparameter_usage: ClassVar[Dict[str, Set[str]]] = defaultdict(set)

    #: Keep track of if this is a base model
    _is_base_model: ClassVar[bool]

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]]

    #: The default loss function class
    loss_default: ClassVar[Type[Loss]] = MarginRankingLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Optional[Mapping[str, Any]]] = dict(margin=1.0, reduction='mean')
    #: The instance of the loss
    loss: Loss

    #: The default regularizer class
    regularizer_default: ClassVar[Optional[Type[Regularizer]]] = None
    #: The default parameters for the default regularizer class
    regularizer_default_kwargs: ClassVar[Optional[Mapping[str, Any]]] = None

    def __init__(
        self,
        triples_factory: TriplesFactory,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        automatic_memory_optimization: Optional[bool] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
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
            self.loss = self.loss_default(**(self.loss_default_kwargs or {}))
        else:
            self.loss = loss

        # TODO: Check loss functions that require 1 and -1 as label but only
        self.is_mr_loss: bool = isinstance(self.loss, MarginRankingLoss)
        self.is_nssa_loss: bool = isinstance(self.loss, NSSALoss)

        # The triples factory facilitates access to the dataset.
        self.triples_factory = triples_factory

        '''
        When predict_with_sigmoid is set to True, the sigmoid function is applied to the logits during evaluation and
        also for predictions after training, but has no effect on the training.
        '''
        self.predict_with_sigmoid = predict_with_sigmoid

        # This allows to store the optimized parameters
        self.automatic_memory_optimization = automatic_memory_optimization

    def __init_subclass__(cls, autoreset: bool = True, **kwargs):  # noqa:D105
        cls._is_base_model = not autoreset
        if not cls._is_base_model:
            _track_hyperparameters(cls)
            _add_post_reset_parameters(cls)

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

    def _reset_parameters_(self):  # noqa: D401
        """Reset all parameters of the model in-place."""
        # cf. https://github.com/mberr/ea-sota-comparison/blob/6debd076f93a329753d819ff4d01567a23053720/src/kgm/utils/torch_utils.py#L317-L372   # noqa:E501
        # Make sure that all modules with parameters do have a reset_parameters method.
        uninitialized_parameters = set(map(id, self.parameters()))
        parents = defaultdict(list)

        # Recursively visit all sub-modules
        task_list = []
        for name, module in self.named_modules():

            # skip self
            if module is self:
                continue

            # Track parents for blaming
            for p in module.parameters():
                parents[id(p)].append(module)

            # call reset_parameters if possible
            if hasattr(module, 'reset_parameters'):
                task_list.append((name.count('.'), module))

        # initialize from bottom to top
        # This ensures that specialized initializations will take priority over the default ones of its components.
        for module in map(itemgetter(1), sorted(task_list, reverse=True, key=itemgetter(0))):
            module.reset_parameters()
            uninitialized_parameters.difference_update(map(id, module.parameters()))

        # emit warning if there where parameters which were not initialised by reset_parameters.
        if len(uninitialized_parameters) > 0:
            logger.warning(
                'reset_parameters() not found for all modules containing parameters. '
                '%d parameters where likely not initialized.',
                len(uninitialized_parameters),
            )

            # Additional debug information
            for i, p_id in enumerate(uninitialized_parameters, start=1):
                logger.debug('[%3d] Parents to blame: %s', i, parents.get(p_id))

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

    def _instantiate_default_regularizer(self, **kwargs) -> Optional[Regularizer]:
        """Instantiate the regularizer from this class's default settings.

        If the default regularizer is None, None is returned.
        Handles the corner case when the default regularizer's keyword arguments are None
        Additional keyword arguments can be passed through to the `__init__()` function
        """
        if self.regularizer_default is None:
            return None

        _kwargs = dict(self.regularizer_default_kwargs or {})
        _kwargs.update(kwargs)
        return self.regularizer_default(**_kwargs)

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
        scores = self.score_t(hr_batch, slice_size=slice_size)  # type: ignore
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
        scores = self.score_r(ht_batch, slice_size=slice_size)  # type: ignore
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
        scores = self.score_t(rt_batch_cloned, slice_size=slice_size)  # type: ignore

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
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "post_parameter_update"):
                module.post_parameter_update()

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
        return self.loss(positive_scores, negative_scores, y) + collect_regularization_terms(self)

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
        return self.loss(tensor_1, tensor_2) + collect_regularization_terms(self)

    @abstractmethod
    def forward(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
        slice_size: Optional[int] = None,
        slice_dim: Optional[str] = None,
    ) -> torch.FloatTensor:
        """Forward pass.

        This method takes head, relation and tail indices and calculates the corresponding score.

        .. note ::
            All indices which are not None, have to be either 1-element or have the same shape, which is the batch size.

        .. note ::
            If slicing is requested, the corresponding indices have to be None.

        :param h_indices:
            The head indices. None indicates to use all.
        :param r_indices:
            The relation indices. None indicates to use all.
        :param t_indices:
            The tail indices. None indicates to use all.
        :param slice_size:
            The slice size.
        :param slice_dim:
            The dimension along which to slice. From {"h", "r", "t"}.

        :return: shape: (batch_size, num_heads, num_relations, num_tails)
            The score for each triple.
        """
        raise NotImplementedError

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass.

        This method takes head, relation and tail of each triple and calculates the corresponding score.

        :param hrt_batch: shape: (batch_size, 3), dtype: long
            The indices of (head, relation, tail) triples.

        :return: shape: (batch_size, 1), dtype: float
            The score for each triple.
        """
        return self(
            h_indices=hrt_batch[:, 0],
            r_indices=hrt_batch[:, 1],
            t_indices=hrt_batch[:, 2],
        ).view(hrt_batch.shape[0], 1)

    def score_t(self, hr_batch: torch.LongTensor, slice_size: Optional[int] = None) -> torch.FloatTensor:
        """Forward pass using right side (tail) prediction.

        This method calculates the score for all possible tails for each (head, relation) pair.

        :param hr_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, relation) pairs.
        :param slice_size:
            The slice size.

        :return: shape: (batch_size, num_entities), dtype: float
            For each h-r pair, the scores for all possible tails.
        """
        return self(
            h_indices=hr_batch[:, 0],
            r_indices=hr_batch[:, 1],
            t_indices=None,
            slice_size=slice_size,
            slice_dim="h",
        ).view(hr_batch.shape[0], self.num_entities)

    def score_h(self, rt_batch: torch.LongTensor, slice_size: Optional[int] = None) -> torch.FloatTensor:
        """Forward pass using left side (head) prediction.

        This method calculates the score for all possible heads for each (relation, tail) pair.

        :param rt_batch: shape: (batch_size, 2), dtype: long
            The indices of (relation, tail) pairs.
        :param slice_size:
            The slice size.

        :return: shape: (batch_size, num_entities), dtype: float
            For each r-t pair, the scores for all possible heads.
        """
        return self(
            h_indices=None,
            r_indices=rt_batch[:, 0],
            t_indices=rt_batch[:, 1],
            slice_size=slice_size,
            slice_dim="r",
        ).view(rt_batch.shape[0], self.num_entities)

    def score_r(self, ht_batch: torch.LongTensor, slice_size: Optional[int] = None) -> torch.FloatTensor:
        """Forward pass using middle (relation) prediction.

        This method calculates the score for all possible relations for each (head, tail) pair.

        :param ht_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, tail) pairs.
        :param slice_size:
            The slice size.

        :return: shape: (batch_size, num_relations), dtype: float
            For each h-t pair, the scores for all possible relations.
        """
        return self(
            h_indices=ht_batch[:, 0],
            r_indices=None,
            t_indices=ht_batch[:, 1],
            slice_size=slice_size,
            slice_dim="t",
        ).view(ht_batch.shape[0], self.num_relations)

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


def _prepare_representation_module_list(
    representations: Union[None, RepresentationModule, Sequence[RepresentationModule]],
) -> Sequence[RepresentationModule]:
    """Normalize list of representations and wrap into nn.ModuleList."""
    # Important: use ModuleList to ensure that Pytorch correctly handles their devices and parameters
    if representations is not None and not isinstance(representations, Sequence):
        representations = [representations]
    return nn.ModuleList(representations)


class ERModel(Model, Generic[HeadRepresentation, RelationRepresentation, TailRepresentation], autoreset=False):
    """A commonly useful base for KGEMs using embeddings and interaction modules."""

    #: The entity representations
    entity_representations: Sequence[RepresentationModule]

    #: The relation representations
    relation_representations: Sequence[RepresentationModule]

    #: The weight regularizers
    weight_regularizers: List[Regularizer]

    def __init__(
        self,
        triples_factory: TriplesFactory,
        interaction: Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation],
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        automatic_memory_optimization: Optional[bool] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        entity_representations: Union[None, RepresentationModule, Sequence[RepresentationModule]] = None,
        relation_representations: Union[None, RepresentationModule, Sequence[RepresentationModule]] = None,
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
        """
        super().__init__(
            triples_factory=triples_factory,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            predict_with_sigmoid=predict_with_sigmoid,
        )
        self.entity_representations = _prepare_representation_module_list(representations=entity_representations)
        self.relation_representations = _prepare_representation_module_list(representations=relation_representations)
        self.interaction = interaction
        # Comment: it is important that the regularizers are stored in a module list, in order to appear in
        # model.modules(). Thereby, we can collect them automatically.
        self.weight_regularizers = nn.ModuleList()

    def append_weight_regularizer(
        self,
        parameter: Union[str, nn.Parameter, Iterable[Union[str, nn.Parameter]]],
        regularizer: Regularizer,
    ) -> None:
        """Add a model weight to a regularizer's weight list, and register the regularizer with the model.

        :param parameter:
            The parameter, either as name, or as nn.Parameter object. A list of available parameter names is shown by
             `sorted(dict(self.named_parameters()).keys())`.
        :param regularizer:
            The regularizer instance which will regularize the weights.
        """
        # normalize input
        if isinstance(parameter, (str, nn.Parameter)):
            parameter = [parameter]
        weights: Mapping[str, nn.Parameter] = dict(self.named_parameters())
        for param in parameter:
            if isinstance(param, str):
                if parameter not in weights.keys():
                    raise ValueError(f"Invalid parameter_name={parameter}. Available are: {sorted(weights.keys())}.")
                param: nn.Parameter = weights[param]  # type: ignore
            regularizer.add_parameter(parameter=param)
        self.weight_regularizers.append(regularizer)

    def forward(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
        slice_size: Optional[int] = None,
        slice_dim: Optional[str] = None,
    ) -> torch.FloatTensor:
        """Forward pass.

        This method takes head, relation and tail indices and calculates the corresponding score.

        All indices which are not None, have to be either 1-element or have the same shape, which is the batch size.

        :param h_indices:
            The head indices. None indicates to use all.
        :param r_indices:
            The relation indices. None indicates to use all.
        :param t_indices:
            The tail indices. None indicates to use all.
        :param slice_size:
            The slice size.
        :param slice_dim:
            The dimension along which to slice. From {"h", "r", "t"}

        :return: shape: (batch_size, num_heads, num_relations, num_tails)
            The score for each triple.
        """
        h, r, t = self._get_representations(h_indices, r_indices, t_indices)
        scores = self.interaction.score(h=h, r=r, t=t, slice_size=slice_size, slice_dim=slice_dim)
        return self._repeat_scores_if_necessary(scores, h_indices, r_indices, t_indices)

    def _repeat_scores_if_necessary(
        self,
        scores: torch.FloatTensor,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
    ) -> torch.FloatTensor:
        repeat_relations = len(self.relation_representations) == 0
        repeat_entities = len(self.entity_representations) == 0

        if not (repeat_entities or repeat_relations):
            return scores

        repeats = [1, 1, 1, 1]

        for i, (flag, ind, num) in enumerate((
            (repeat_entities, h_indices, self.num_entities),
            (repeat_relations, r_indices, self.num_relations),
            (repeat_entities, t_indices, self.num_entities),
        ), start=1):
            if flag:
                if ind is None:
                    repeats[i] = num
                else:
                    batch_size = len(ind)
                    if scores.shape[0] < batch_size:
                        repeats[0] = batch_size

        return scores.repeat(*repeats)

    def _get_representations(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
    ) -> Tuple[
        Union[torch.FloatTensor, Sequence[torch.FloatTensor]],
        Union[torch.FloatTensor, Sequence[torch.FloatTensor]],
        Union[torch.FloatTensor, Sequence[torch.FloatTensor]],
    ]:
        h, r, t = [
            [
                representation.get_in_canonical_shape(indices=indices)
                for representation in representations
            ]
            for indices, representations in (
                (h_indices, self.entity_representations),
                (r_indices, self.relation_representations),
                (t_indices, self.entity_representations),
            )
        ]
        # normalization
        h, r, t = [x[0] if len(x) == 1 else x for x in (h, r, t)]
        return h, r, t


class SingleVectorEmbeddingModel(ERModel, autoreset=False):
    """A KGEM that stores one :class:`pykeen.nn.Embedding` for each entities and relations."""

    def __init__(
        self,
        triples_factory: TriplesFactory,
        interaction: Interaction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
        embedding_dim: int = 200,
        relation_dim: Union[None, int, Sequence[int]] = None,
        automatic_memory_optimization: Optional[bool] = None,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        embedding_specification: Optional[EmbeddingSpecification] = None,
        relation_embedding_specification: Optional[EmbeddingSpecification] = None,
    ) -> None:
        """Initialize embedding model.

        :param triples_factory:
            The triple factory connected to the model.
        :param interaction:
            The embedding-based interaction function used to compute scores.
        :param embedding_dim:
            The embedding dimensionality of the entity embeddings.
        :param automatic_memory_optimization:
            Whether to automatically optimize the sub-batch size during training and batch size during evaluation with
            regards to the hardware at hand.
        :param loss:
            The loss to use.
        :param preferred_device:
            The default device where to model is located.
        :param random_seed:
            An optional random seed to set before the initialization of weights.
        """
        # Default for relation dimensionality
        if relation_dim is None:
            relation_dim = embedding_dim
        super().__init__(
            triples_factory=triples_factory,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            preferred_device=preferred_device,
            random_seed=random_seed,
            interaction=interaction,
            entity_representations=Embedding.from_specification(
                num_embeddings=triples_factory.num_entities,
                shape=embedding_dim,
                specification=embedding_specification,
            ),
            relation_representations=Embedding.from_specification(
                num_embeddings=triples_factory.num_relations,
                shape=relation_dim,
                specification=relation_embedding_specification,
            ),
        )

    @property
    def embedding_dim(self) -> int:  # noqa:D401
        """The entity embedding dim."""
        # TODO: Deprecated; directly use self.entity_representations[0].embedding_dim instead?
        embedding = self.entity_representations[0]
        assert isinstance(embedding, Embedding)
        return embedding.embedding_dim


class DoubleRelationEmbeddingModel(ERModel, autoreset=False):
    """A KGEM that stores one :class:`pykeen.nn.Embedding` for entities and two for relations.

    .. seealso::

        - :class:`pykeen.models.StructuredEmbedding`
        - :class:`pykeen.models.TransH`
    """

    def __init__(
        self,
        triples_factory: TriplesFactory,
        interaction: Interaction[
            torch.FloatTensor,
            Tuple[torch.FloatTensor, torch.FloatTensor],
            torch.FloatTensor,
        ],
        embedding_dim: int = 50,
        relation_dim: Union[None, int, Sequence[int]] = None,
        second_relation_dim: Union[None, int, Sequence[int]] = None,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        automatic_memory_optimization: Optional[bool] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        embedding_specification: Optional[EmbeddingSpecification] = None,
        relation_embedding_specification: Optional[EmbeddingSpecification] = None,
        second_relation_embedding_specification: Optional[EmbeddingSpecification] = None,
    ) -> None:
        if relation_dim is None:
            relation_dim = embedding_dim
        if second_relation_dim is None:
            second_relation_dim = relation_dim
        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            preferred_device=preferred_device,
            random_seed=random_seed,
            entity_representations=[
                Embedding.from_specification(
                    num_embeddings=triples_factory.num_entities,
                    shape=embedding_dim,
                    specification=embedding_specification,
                ),
            ],
            relation_representations=[
                Embedding.from_specification(
                    num_embeddings=triples_factory.num_relations,
                    shape=relation_dim,
                    specification=relation_embedding_specification,
                ),
                Embedding.from_specification(
                    num_embeddings=triples_factory.num_relations,
                    shape=second_relation_dim,
                    specification=second_relation_embedding_specification,
                ),
            ],
        )


class TwoVectorEmbeddingModel(ERModel, autoreset=False):
    """A KGEM that stores two :class:`pykeen.nn.Embedding` for each entities and relations.

    .. seealso::

        - :class:`pykeen.models.KG2E`
        - :class:`pykeen.models.TransD`
    """

    def __init__(
        self,
        triples_factory: TriplesFactory,
        interaction: Interaction[
            Tuple[torch.FloatTensor, torch.FloatTensor],
            Tuple[torch.FloatTensor, torch.FloatTensor],
            Tuple[torch.FloatTensor, torch.FloatTensor],
        ],
        embedding_dim: int = 50,
        relation_dim: Optional[int] = None,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        automatic_memory_optimization: Optional[bool] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        embedding_specification: Optional[EmbeddingSpecification] = None,
        relation_embedding_specification: Optional[EmbeddingSpecification] = None,
        second_embedding_specification: Optional[EmbeddingSpecification] = None,
        second_relation_embedding_specification: Optional[EmbeddingSpecification] = None,
    ) -> None:
        if relation_dim is None:
            relation_dim = embedding_dim
        super().__init__(
            triples_factory=triples_factory,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            preferred_device=preferred_device,
            random_seed=random_seed,
            entity_representations=[
                Embedding.from_specification(
                    num_embeddings=triples_factory.num_entities,
                    embedding_dim=embedding_dim,
                    specification=embedding_specification,
                ),
                Embedding.from_specification(
                    num_embeddings=triples_factory.num_entities,
                    embedding_dim=embedding_dim,
                    specification=second_embedding_specification,
                ),
            ],
            relation_representations=[
                Embedding.from_specification(
                    num_embeddings=triples_factory.num_relations,
                    embedding_dim=relation_dim,
                    specification=relation_embedding_specification,
                ),
                Embedding.from_specification(
                    num_embeddings=triples_factory.num_relations,
                    embedding_dim=relation_dim,
                    specification=second_relation_embedding_specification,
                ),
            ],
            interaction=interaction,
        )


class TwoSideEmbeddingModel(ERModel, autoreset=False):
    """A KGEM with two sub-KGEMs that serve as a "forwards" and "backwards" model.

    Stores two :class:`pykeen.nn.Embedding` for each entities and relations.

    .. seealso:: :class:`pykeen.models.SimplE`
    """

    def __init__(
        self,
        triples_factory: TriplesFactory,
        interaction: Interaction,
        embedding_dim: int = 50,
        relation_dim: Optional[int] = None,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        automatic_memory_optimization: Optional[bool] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        embedding_specification: Optional[EmbeddingSpecification] = None,
        relation_embedding_specification: Optional[EmbeddingSpecification] = None,
        second_embedding_specification: Optional[EmbeddingSpecification] = None,
        second_relation_embedding_specification: Optional[EmbeddingSpecification] = None,
    ):
        if relation_dim is None:
            relation_dim = embedding_dim
        super().__init__(
            triples_factory=triples_factory,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            preferred_device=preferred_device,
            random_seed=random_seed,
            entity_representations=[
                Embedding.from_specification(
                    num_embeddings=triples_factory.num_entities,
                    embedding_dim=embedding_dim,
                    specification=embedding_specification,
                ),
                Embedding.from_specification(
                    num_embeddings=triples_factory.num_entities,
                    embedding_dim=embedding_dim,
                    specification=second_embedding_specification,
                ),
            ],
            relation_representations=[
                Embedding.from_specification(
                    num_embeddings=triples_factory.num_relations,
                    embedding_dim=relation_dim,
                    specification=relation_embedding_specification,
                ),
                Embedding.from_specification(
                    num_embeddings=triples_factory.num_relations,
                    embedding_dim=relation_dim,
                    specification=second_relation_embedding_specification,
                ),
            ],
            interaction=interaction,
        )

    def forward(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
        slice_size: Optional[int] = None,
        slice_dim: Optional[str] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        return 0.5 * sum(
            self.interaction.score(
                h_source.get_in_canonical_shape(indices=h_indices),
                r_source.get_in_canonical_shape(indices=r_indices),
                t_source.get_in_canonical_shape(indices=t_indices),
                slice_size=slice_size,
                slice_dim=slice_dim,
            )
            for h_source, r_source, t_source in (
                (self.entity_representations[0], self.relation_representations[0], self.entity_representations[1]),
                (self.entity_representations[1], self.relation_representations[1], self.entity_representations[0]),
            )
        )
