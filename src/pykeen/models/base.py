# -*- coding: utf-8 -*-

"""Base module for all KGE models."""

from __future__ import annotations

import functools
import inspect
import logging
import os
import pickle
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, ClassVar, Generic, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union, cast

import pandas as pd
import torch
from _operator import itemgetter
from class_resolver import HintOrType, OneOrManyHintOrType, OneOrManyOptionalKwargs, OptionalKwargs
from docdata import parse_docdata
from torch import nn

from ..losses import Loss, MarginRankingLoss, loss_resolver
from ..nn import Interaction, Representation, interaction_resolver, representation_resolver
from ..nn.representation import Representation, build_representation
from ..regularizers import NoRegularizer, Regularizer, regularizer_resolver
from ..triples import KGInfo, relation_inverter
from ..typing import (
    LABEL_HEAD,
    LABEL_RELATION,
    LABEL_TAIL,
    HeadRepresentation,
    InductiveMode,
    MappedTriples,
    RelationRepresentation,
    ScorePack,
    TailRepresentation,
    Target,
)
from ..utils import (
    NoRandomSeedNecessary,
    check_shapes,
    extend_batch,
    get_batchnorm_modules,
    get_preferred_device,
    set_random_seed,
)

__all__ = [
    "Model",
    "_OldAbstractModel",
    "EntityRelationEmbeddingModel",
]

logger = logging.getLogger(__name__)


class Model(nn.Module, ABC):
    """A base module for KGE models.

    Subclasses of :class:`Model` can decide however they want on how to store entities' and
    relations' representations, how they want to be looked up, and how they should
    be scored. The :class:`OModel` provides a commonly used interface for models storing entity
    and relation representations in the form of :class:`pykeen.nn.Embedding`.
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]]

    _random_seed: Optional[int]

    #: The default loss function class
    loss_default: ClassVar[Type[Loss]] = MarginRankingLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Optional[Mapping[str, Any]]] = dict(margin=1.0, reduction="mean")
    #: The instance of the loss
    loss: Loss

    num_entities: int
    num_relations: int
    use_inverse_triples: bool

    can_slice_h: ClassVar[bool]
    can_slice_r: ClassVar[bool]
    can_slice_t: ClassVar[bool]

    def __init__(
        self,
        *,
        triples_factory: KGInfo,
        loss: HintOrType[Loss] = None,
        loss_kwargs: Optional[Mapping[str, Any]] = None,
        predict_with_sigmoid: bool = False,
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
        :param random_seed:
            A random seed to use for initialising the model's weights. **Should** be set when aiming at reproducibility.
        """
        super().__init__()

        # Random seeds have to set before the embeddings are initialized
        if random_seed is None:
            logger.warning("No random seed is specified. This may lead to non-reproducible results.")
            self._random_seed = None
        elif random_seed is not NoRandomSeedNecessary:
            set_random_seed(random_seed)
            self._random_seed = random_seed

        # Loss
        if loss is None:
            self.loss = self.loss_default(**(self.loss_default_kwargs or {}))
        else:
            self.loss = loss_resolver.make(loss, pos_kwargs=loss_kwargs)

        self.use_inverse_triples = triples_factory.create_inverse_triples
        self.num_entities = triples_factory.num_entities
        self.num_relations = triples_factory.num_relations

        """
        When predict_with_sigmoid is set to True, the sigmoid function is applied to the logits during evaluation and
        also for predictions after training, but has no effect on the training.
        """
        self.predict_with_sigmoid = predict_with_sigmoid

    @property
    def num_real_relations(self) -> int:
        """Return the real number of relations (without inverses)."""
        if self.use_inverse_triples:
            return self.num_relations // 2
        return self.num_relations

    def __init_subclass__(cls, **kwargs):
        """Initialize the subclass.

        This checks for all subclasses if they are tagged with :class:`abc.ABC` with :func:`inspect.isabstract`.
        All non-abstract deriving models should have citation information. Subclasses can further override
        ``__init_subclass__``, but need to remember to call ``super().__init_subclass__`` as well so this
        gets run.
        """
        if not inspect.isabstract(cls):
            parse_docdata(cls)

    @property
    def device(self) -> torch.device:
        """Return the model's device."""
        return get_preferred_device(self, allow_ambiguity=False)

    def reset_parameters_(self):  # noqa: D401
        """Reset all parameters of the model and enforce model constraints."""
        self._reset_parameters_()
        # TODO: why do we need to empty the cache?
        torch.cuda.empty_cache()
        self.post_parameter_update()
        return self

    """Base methods"""

    def post_forward_pass(self):
        """Run after calculating the forward loss."""

    def _free_graph_and_cache(self):
        """Run to free the graph and cache."""

    """Abstract methods"""

    @abstractmethod
    def _reset_parameters_(self):  # noqa: D401
        """Reset all parameters of the model in-place."""

    @abstractmethod
    def _get_entity_len(self, *, mode: Optional[InductiveMode]) -> Optional[int]:
        """Get the number of entities depending on the mode parameters."""

    def post_parameter_update(self) -> None:
        """Has to be called after each parameter update."""

    """Abstract methods - Scoring"""

    @abstractmethod
    def score_hrt(self, hrt_batch: torch.LongTensor, *, mode: Optional[InductiveMode] = None) -> torch.FloatTensor:
        """Forward pass.

        This method takes head, relation and tail of each triple and calculates the corresponding score.

        :param hrt_batch: shape: (batch_size, 3), dtype: long
            The indices of (head, relation, tail) triples.
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.
        :return: shape: (batch_size, 1), dtype: float
            The score for each triple.
        """

    @abstractmethod
    def score_t(
        self, hr_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:
        """Forward pass using right side (tail) prediction.

        This method calculates the score for all possible tails for each (head, relation) pair.

        :param hr_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, relation) pairs.
        :param slice_size: >0
            The divisor for the scoring function when using slicing.
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.

        :return: shape: (batch_size, num_entities), dtype: float
            For each h-r pair, the scores for all possible tails.
        """

    @abstractmethod
    def score_r(
        self, ht_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:
        """Forward pass using middle (relation) prediction.

        This method calculates the score for all possible relations for each (head, tail) pair.

        :param ht_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, tail) pairs.
        :param slice_size: >0
            The divisor for the scoring function when using slicing.
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.

        :return: shape: (batch_size, num_real_relations), dtype: float
            For each h-t pair, the scores for all possible relations.
        """
        # TODO: this currently compute (batch_size, num_relations) instead,
        # i.e., scores for normal and inverse relations

    @abstractmethod
    def score_h(
        self, rt_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:
        """Forward pass using left side (head) prediction.

        This method calculates the score for all possible heads for each (relation, tail) pair.

        :param rt_batch: shape: (batch_size, 2), dtype: long
            The indices of (relation, tail) pairs.
        :param slice_size: >0
            The divisor for the scoring function when using slicing.
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.

        :return: shape: (batch_size, num_entities), dtype: float
            For each r-t pair, the scores for all possible heads.
        """

    @abstractmethod
    def collect_regularization_term(self) -> torch.FloatTensor:
        """Get the regularization term for the loss function."""

    """Concrete methods"""

    def get_grad_params(self) -> Iterable[nn.Parameter]:
        """Get the parameters that require gradients."""
        # TODO: Why do we need that? The optimizer takes care of filtering the parameters.
        return filter(lambda p: p.requires_grad, self.parameters())

    @property
    def num_parameter_bytes(self) -> int:
        """Calculate the number of bytes used for all parameters of the model."""
        return sum(param.numel() * param.element_size() for param in self.parameters(recurse=True))

    @property
    def num_parameters(self) -> int:
        """Calculate the number of parameters of the model."""
        return sum(param.numel() for param in self.parameters(recurse=True))

    def save_state(self, path: Union[str, os.PathLike]) -> None:
        """Save the state of the model.

        :param path:
            Path of the file where to store the state in.
        """
        torch.save(self.state_dict(), path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, path: Union[str, os.PathLike]) -> None:
        """Load the state of the model.

        :param path:
            Path of the file where to load the state from.
        """
        self.load_state_dict(torch.load(path, map_location=self.device))

    """Prediction methods"""

    def _prepare_batch(self, batch: torch.LongTensor, index_relation: int) -> torch.LongTensor:
        # send to device
        batch = batch.to(self.device)

        # special handling of inverse relations
        if not self.use_inverse_triples:
            return batch

        # when trained on inverse relations, the internal relation ID is twice the original relation ID
        return relation_inverter.map(batch=batch, index=index_relation, invert=False)

    def predict_hrt(self, hrt_batch: torch.LongTensor, *, mode: Optional[InductiveMode] = None) -> torch.FloatTensor:
        """Calculate the scores for triples.

        This method takes head, relation and tail of each triple and calculates the corresponding score.

        Additionally, the model is set to evaluation mode.

        :param hrt_batch: shape: (number of triples, 3), dtype: long
            The indices of (head, relation, tail) triples.
        :param mode:
            The pass mode. Is None for transductive and "training" / "validation" / "testing" in inductive.

        :return: shape: (number of triples, 1), dtype: float
            The score for each triple.
        """
        self.eval()  # Enforce evaluation mode
        scores = self.score_hrt(self._prepare_batch(batch=hrt_batch, index_relation=1), mode=mode)
        if self.predict_with_sigmoid:
            scores = torch.sigmoid(scores)
        return scores

    def predict_h(
        self, rt_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:
        """Forward pass using left side (head) prediction for obtaining scores of all possible heads.

        This method calculates the score for all possible heads for each (relation, tail) pair.

        .. note::

            If the model has been trained with inverse relations, the task of predicting
            the head entities becomes the task of predicting the tail entities of the
            inverse triples, i.e., $f(*,r,t)$ is predicted by means of $f(t,r_{inv},*)$.

        Additionally, the model is set to evaluation mode.

        :param rt_batch: shape: (batch_size, 2), dtype: long
            The indices of (relation, tail) pairs.
        :param slice_size: >0
            The divisor for the scoring function when using slicing.
        :param mode:
            The pass mode. Is None for transductive and "training" / "validation" / "testing" in inductive.

        :return: shape: (batch_size, num_entities), dtype: float
            For each r-t pair, the scores for all possible heads.
        """
        self.eval()  # Enforce evaluation mode
        rt_batch = self._prepare_batch(batch=rt_batch, index_relation=0)
        if self.use_inverse_triples:
            scores = self.score_h_inverse(rt_batch=rt_batch, slice_size=slice_size, mode=mode)
        else:
            scores = self.score_h(rt_batch, slice_size=slice_size, mode=mode)
        if self.predict_with_sigmoid:
            scores = torch.sigmoid(scores)
        return scores

    def predict_t(
        self,
        hr_batch: torch.LongTensor,
        *,
        slice_size: Optional[int] = None,
        mode: Optional[InductiveMode] = None,
    ) -> torch.FloatTensor:
        """Forward pass using right side (tail) prediction for obtaining scores of all possible tails.

        This method calculates the score for all possible tails for each (head, relation) pair.

        Additionally, the model is set to evaluation mode.

        :param hr_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, relation) pairs.
        :param slice_size: >0
            The divisor for the scoring function when using slicing.
        :param mode:
            The pass mode. Is None for transductive and "training" / "validation" / "testing" in inductive.

        :return: shape: (batch_size, num_entities), dtype: float
            For each h-r pair, the scores for all possible tails.

        .. note::

            We only expect the right side-predictions, i.e., $(h,r,*)$ to change its
            default behavior when the model has been trained with inverse relations
            (mainly because of the behavior of the LCWA training approach). This is why
            the :func:`predict_h` has different behavior depending on
            if inverse triples were used in training, and why this function has the same
            behavior regardless of the use of inverse triples.
        """
        self.eval()  # Enforce evaluation mode
        hr_batch = self._prepare_batch(batch=hr_batch, index_relation=1)
        scores = self.score_t(hr_batch, slice_size=slice_size, mode=mode)
        if self.predict_with_sigmoid:
            scores = torch.sigmoid(scores)
        return scores

    def predict_r(
        self,
        ht_batch: torch.LongTensor,
        *,
        slice_size: Optional[int] = None,
        mode: Optional[InductiveMode] = None,
    ) -> torch.FloatTensor:
        """Forward pass using middle (relation) prediction for obtaining scores of all possible relations.

        This method calculates the score for all possible relations for each (head, tail) pair.

        Additionally, the model is set to evaluation mode.

        :param ht_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, tail) pairs.
        :param slice_size: >0
            The divisor for the scoring function when using slicing.
        :param mode:
            The pass mode. Is None for transductive and "training" / "validation" / "testing" in inductive.

        :return: shape: (batch_size, num_real_relations), dtype: float
            For each h-t pair, the scores for all possible relations.
        """
        self.eval()  # Enforce evaluation mode
        ht_batch = ht_batch.to(self.device)
        scores = self.score_r(ht_batch, slice_size=slice_size, mode=mode)
        if self.predict_with_sigmoid:
            scores = torch.sigmoid(scores)
        return scores

    def predict(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        *,
        slice_size: Optional[int] = None,
        mode: Optional[InductiveMode],
    ) -> torch.FloatTensor:
        """Predict scores for the given target."""
        if target == LABEL_TAIL:
            return self.predict_t(hrt_batch[:, 0:2], slice_size=slice_size, mode=mode)

        if target == LABEL_RELATION:
            return self.predict_r(hrt_batch[:, [0, 2]], slice_size=slice_size, mode=mode)

        if target == LABEL_HEAD:
            return self.predict_h(hrt_batch[:, 1:3], slice_size=slice_size, mode=mode)

        raise ValueError(f"Unknown target={target}")

    def get_all_prediction_df(
        self,
        *,
        k: Optional[int] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> Union[ScorePack, pd.DataFrame]:
        """Compute scores for all triples, optionally returning only the k highest scoring.

        .. note:: This operation is computationally very expensive for reasonably-sized knowledge graphs.
        .. warning:: Setting k=None may lead to huge memory requirements.

        :param k:
            The number of triples to return. Set to None, to keep all.
        :param batch_size:
            The batch size to use for calculating scores.
        :param kwargs: Additional kwargs to pass to :func:`pykeen.models.predict.get_all_prediction_df`.
        :return: shape: (k, 3)
            A tensor containing the k highest scoring triples, or all possible triples if k=None.
        """
        from .predict import get_all_prediction_df

        warnings.warn("Use pykeen.models.predict.get_all_prediction_df", DeprecationWarning)
        return get_all_prediction_df(model=self, k=k, batch_size=batch_size, **kwargs)

    def get_head_prediction_df(
        self,
        relation_label: str,
        tail_label: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Predict heads for the given relation and tail (given by label).

        :param relation_label: The string label for the relation
        :param tail_label: The string label for the tail entity
        :param kwargs: Keyword arguments passed to :func:`pykeen.models.predict.get_head_prediction_df`

        The following example shows that after you train a model on the Nations dataset,
        you can score all entities w.r.t a given relation and tail entity.

        >>> from pykeen.pipeline import pipeline
        >>> result = pipeline(
        ...     dataset='Nations',
        ...     model='RotatE',
        ... )
        >>> df = result.model.get_head_prediction_df('accusation', 'brazil', triples_factory=result.training)
        """
        from .predict import get_head_prediction_df

        warnings.warn("Use pykeen.models.predict.get_head_prediction_df", DeprecationWarning)
        return get_head_prediction_df(self, relation_label=relation_label, tail_label=tail_label, **kwargs)

    def get_relation_prediction_df(
        self,
        head_label: str,
        tail_label: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Predict relations for the given head and tail (given by label).

        :param head_label: The string label for the head entity
        :param tail_label: The string label for the tail entity
        :param kwargs: Keyword arguments passed to :func:`pykeen.models.predict.get_relation_prediction_df`
        """
        from .predict import get_relation_prediction_df

        warnings.warn("Use pykeen.models.predict.get_relation_prediction_df", DeprecationWarning)
        return get_relation_prediction_df(self, head_label=head_label, tail_label=tail_label, **kwargs)

    def get_tail_prediction_df(
        self,
        head_label: str,
        relation_label: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Predict tails for the given head and relation (given by label).

        :param head_label: The string label for the head entity
        :param relation_label: The string label for the relation
        :param kwargs: Keyword arguments passed to :func:`pykeen.models.predict.get_tail_prediction_df`

        The following example shows that after you train a model on the Nations dataset,
        you can score all entities w.r.t a given head entity and relation.

        >>> from pykeen.pipeline import pipeline
        >>> result = pipeline(
        ...     dataset='Nations',
        ...     model='RotatE',
        ... )
        >>> df = result.model.get_tail_prediction_df('brazil', 'accusation', triples_factory=result.training)
        """
        from .predict import get_tail_prediction_df

        warnings.warn("Use pykeen.models.predict.get_tail_prediction_df", DeprecationWarning)
        return get_tail_prediction_df(self, head_label=head_label, relation_label=relation_label, **kwargs)

    """Inverse scoring"""

    def _prepare_inverse_batch(self, batch: torch.LongTensor, index_relation: int) -> torch.LongTensor:
        if not self.use_inverse_triples:
            raise ValueError(
                "Your model is not configured to predict with inverse relations."
                " Set ``create_inverse_triples=True`` when creating the dataset/triples factory"
                " or using the pipeline().",
            )
        return relation_inverter.invert_(batch=batch, index=index_relation).flip(1)

    def score_hrt_inverse(
        self,
        hrt_batch: torch.LongTensor,
        *,
        mode: Optional[InductiveMode],
    ) -> torch.FloatTensor:
        r"""Score triples based on inverse triples, i.e., compute $f(h,r,t)$ based on $f(t,r_{inv},h)$.

        When training with inverse relations, the model produces two (different) scores for a triple $(h,r,t) \in K$.
        The forward score is calculated from $f(h,r,t)$ and the inverse score is calculated from $f(t,r_{inv},h)$.
        This function enables users to inspect the scores obtained by using the corresponding inverse triples.
        """
        t_r_inv_h = self._prepare_inverse_batch(batch=hrt_batch, index_relation=1)
        return self.score_hrt(hrt_batch=t_r_inv_h, mode=mode)

    def score_t_inverse(
        self, hr_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode]
    ):
        """Score all tails for a batch of (h,r)-pairs using the head predictions for the inverses $(*,r_{inv},h)$."""
        r_inv_h = self._prepare_inverse_batch(batch=hr_batch, index_relation=1)
        return self.score_h(rt_batch=r_inv_h, slice_size=slice_size, mode=mode)

    def score_h_inverse(
        self, rt_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode]
    ):
        """Score all heads for a batch of (r,t)-pairs using the tail predictions for the inverses $(t,r_{inv},*)$."""
        t_r_inv = self._prepare_inverse_batch(batch=rt_batch, index_relation=0)
        return self.score_t(hr_batch=t_r_inv, slice_size=slice_size, mode=mode)


class _OldAbstractModel(Model, ABC, autoreset=False):
    """A base module for PyKEEN 1.0-style KGE models."""

    #: The default regularizer class
    regularizer_default: ClassVar[Optional[Type[Regularizer]]] = None
    #: The default parameters for the default regularizer class
    regularizer_default_kwargs: ClassVar[Optional[Mapping[str, Any]]] = None
    #: The instance of the regularizer
    regularizer: Regularizer  # type: ignore

    can_slice_h = False
    can_slice_r = False
    can_slice_t = False

    def __init__(
        self,
        *,
        triples_factory: KGInfo,
        regularizer: Optional[Regularizer] = None,
        **kwargs,
    ) -> None:
        """Initialize the module.

        :param triples_factory:
            The triples factory facilitates access to the dataset.
        :param regularizer:
            A regularizer to use for training.
        :param kwargs:
            additional keyword-based arguments passed to Model.__init__
        """
        super().__init__(triples_factory=triples_factory, **kwargs)
        # Regularizer
        if regularizer is not None:
            self.regularizer = regularizer
        elif self.regularizer_default is not None:
            self.regularizer = self.regularizer_default(
                **(self.regularizer_default_kwargs or {}),
            )
        else:
            self.regularizer = NoRegularizer()

    def __init_subclass__(cls, autoreset: bool = True, **kwargs):  # noqa:D105
        super().__init_subclass__(**kwargs)
        if autoreset:
            _add_post_reset_parameters(cls)

    def _get_entity_len(self, mode: Optional[InductiveMode] = None) -> int:  # noqa:D105
        if mode is not None:
            raise ValueError
        return self.num_entities

    def post_parameter_update(self) -> None:
        """Has to be called after each parameter update."""
        self.regularizer.reset()

    def regularize_if_necessary(self, *tensors: torch.FloatTensor) -> None:
        """Update the regularizer's term given some tensors, if regularization is requested.

        :param tensors: The tensors that should be passed to the regularizer to update its term.
        """
        if self.training:
            self.regularizer.update(*tensors)

    def score_t(
        self, hr_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:
        """Forward pass using right side (tail) prediction.

        This method calculates the score for all possible tails for each (head, relation) pair.

        :param hr_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, relation) pairs.
        :param slice_size: >0
            The divisor for the scoring function when using slicing.
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.

        :return: shape: (batch_size, num_entities), dtype: float
            For each h-r pair, the scores for all possible tails.
        """
        logger.warning(
            "Calculations will fall back to using the score_hrt method, since this model does not have a specific "
            "score_t function. This might cause the calculations to take longer than necessary.",
        )
        # Extend the hr_batch such that each (h, r) pair is combined with all possible tails
        hrt_batch = extend_batch(batch=hr_batch, max_id=self.num_entities, dim=2)
        # Calculate the scores for each (h, r, t) triple using the generic interaction function
        expanded_scores = self.score_hrt(hrt_batch=hrt_batch, mode=mode)
        # Reshape the scores to match the pre-defined output shape of the score_t function.
        scores = expanded_scores.view(hr_batch.shape[0], -1)
        return scores

    def score_h(
        self, rt_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:
        """Forward pass using left side (head) prediction.

        This method calculates the score for all possible heads for each (relation, tail) pair.

        :param rt_batch: shape: (batch_size, 2), dtype: long
            The indices of (relation, tail) pairs.
        :param slice_size: >0
            The divisor for the scoring function when using slicing.
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.

        :return: shape: (batch_size, num_entities), dtype: float
            For each r-t pair, the scores for all possible heads.
        """
        logger.warning(
            "Calculations will fall back to using the score_hrt method, since this model does not have a specific "
            "score_h function. This might cause the calculations to take longer than necessary.",
        )
        # Extend the rt_batch such that each (r, t) pair is combined with all possible heads
        hrt_batch = extend_batch(batch=rt_batch, max_id=self.num_entities, dim=0)
        # Calculate the scores for each (h, r, t) triple using the generic interaction function
        expanded_scores = self.score_hrt(hrt_batch=hrt_batch, mode=mode)
        # Reshape the scores to match the pre-defined output shape of the score_h function.
        scores = expanded_scores.view(rt_batch.shape[0], -1)
        return scores

    def score_r(
        self, ht_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:
        """Forward pass using middle (relation) prediction.

        This method calculates the score for all possible relations for each (head, tail) pair.

        :param ht_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, tail) pairs.
        :param slice_size: >0
            The divisor for the scoring function when using slicing.
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.

        :return: shape: (batch_size, num_relations), dtype: float
            For each h-t pair, the scores for all possible relations.
        """
        logger.warning(
            "Calculations will fall back to using the score_hrt method, since this model does not have a specific "
            "score_r function. This might cause the calculations to take longer than necessary.",
        )
        # Extend the ht_batch such that each (h, t) pair is combined with all possible relations
        hrt_batch = extend_batch(batch=ht_batch, max_id=self.num_relations, dim=1)
        # Calculate the scores for each (h, r, t) triple using the generic interaction function
        expanded_scores = self.score_hrt(hrt_batch=hrt_batch, mode=mode)
        # Reshape the scores to match the pre-defined output shape of the score_r function.
        scores = expanded_scores.view(ht_batch.shape[0], -1)
        return scores

    # docstr-coverage: inherited
    def collect_regularization_term(self) -> torch.FloatTensor:  # noqa: D102
        return self.regularizer.term

    def post_forward_pass(self):
        """Run after calculating the forward loss."""
        self.regularizer.reset()

    def _free_graph_and_cache(self):
        self.regularizer.reset()


class EntityRelationEmbeddingModel(_OldAbstractModel, ABC, autoreset=False):
    """A base module for KGE models that have different embeddings for entities and relations."""

    #: Primary embeddings for entities
    entity_embeddings: Representation

    #: Primary embeddings for relations
    relation_embeddings: Representation

    def __init__(
        self,
        *,
        triples_factory: KGInfo,
        entity_representations: HintOrType[Representation] = None,
        entity_representations_kwargs: OptionalKwargs = None,
        relation_representations: HintOrType[Representation] = None,
        relation_representations_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        """Initialize the entity embedding model.

        .. seealso:: Constructor of the base class :class:`pykeen.models.Model`
        """
        super().__init__(triples_factory=triples_factory, **kwargs)
        self.entity_embeddings = build_representation(
            max_id=triples_factory.num_entities,
            representation=entity_representations,
            representation_kwargs=entity_representations_kwargs,
        )
        self.relation_embeddings = build_representation(
            max_id=triples_factory.num_relations,
            representation=relation_representations,
            representation_kwargs=relation_representations_kwargs,
        )

    @property
    def embedding_dim(self) -> int:  # noqa:D401
        """The entity embedding dimension."""
        return self.entity_embeddings.embedding_dim

    @property
    def entity_representations(self) -> Sequence[Representation]:  # noqa:D401
        """The entity representations.

        This property provides forward compatibility with the new-style :class:`pykeen.models.ERModel`.
        """
        return [self.entity_embeddings]

    @property
    def relation_representations(self) -> Sequence[Representation]:  # noqa:D401
        """The relation representations.

        This property provides forward compatibility with the new-style :class:`pykeen.models.ERModel`.
        """
        return [self.relation_embeddings]

    # docstr-coverage: inherited
    def _reset_parameters_(self):  # noqa: D102
        self.entity_embeddings.reset_parameters()
        self.relation_embeddings.reset_parameters()

    # docstr-coverage: inherited
    def post_parameter_update(self) -> None:  # noqa: D102
        # make sure to call this first, to reset regularizer state!
        super().post_parameter_update()
        self.entity_embeddings.post_parameter_update()
        self.relation_embeddings.post_parameter_update()


def _add_post_reset_parameters(cls: Type[Model]) -> None:
    # The following lines add in a post-init hook to all subclasses
    # such that the reset_parameters_() function is run
    _original_init = cls.__init__

    @functools.wraps(_original_init)
    def _new_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        self.reset_parameters_()

    # sorry mypy, but this kind of evil must be permitted.
    cls.__init__ = _new_init  # type: ignore


class _NewAbstractModel(Model, ABC):
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

    #: The default regularizer class
    regularizer_default: ClassVar[Optional[Type[Regularizer]]] = None
    #: The default parameters for the default regularizer class
    regularizer_default_kwargs: ClassVar[Optional[Mapping[str, Any]]] = None

    can_slice_h = True
    can_slice_r = True
    can_slice_t = True

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
            if hasattr(module, "reset_parameters"):
                task_list.append((name.count("."), module))

        # initialize from bottom to top
        # This ensures that specialized initializations will take priority over the default ones of its components.
        for module in map(itemgetter(1), sorted(task_list, reverse=True, key=itemgetter(0))):
            module.reset_parameters()
            uninitialized_parameters.difference_update(map(id, module.parameters()))

        # emit warning if there where parameters which were not initialised by reset_parameters.
        if len(uninitialized_parameters) > 0:
            logger.warning(
                "reset_parameters() not found for all modules containing parameters. "
                "%d parameters where likely not initialized.",
                len(uninitialized_parameters),
            )

            # Additional debug information
            for i, p_id in enumerate(uninitialized_parameters, start=1):
                logger.debug("[%3d] Parents to blame: %s", i, parents.get(p_id))

    def _instantiate_regularizer(
        self,
        regularizer: HintOrType[Regularizer],
        regularizer_kwargs: OptionalKwargs = None,
    ) -> Optional[Regularizer]:
        """
        Instantiate a regularizer using the default if None is provided.

        The following precedence order is used:

        1. If the passed regularizer is not None, use it
        2. If the regularizer is None, use the default regularizer. In this case, the
           default kwargs will be used in favor of provided ones.
        3. If both, the regularizer and the default regularizer are None, return None.

        :param regularizer:
            the regularizer, or a hint thereof
        :param regularizer_kwargs:
            additional keyword-based parameters passed to the regularizer upon instantiation

        :return:
            the regularizer instance.
        """
        regularizer, regularizer_kwargs = normalize_with_default(
            choice=regularizer,
            kwargs=regularizer_kwargs,
            default=self.regularizer_default,
            default_kwargs=self.regularizer_default_kwargs,
        )
        return regularizer_resolver.make_safe(regularizer, regularizer_kwargs)

    def post_parameter_update(self) -> None:
        """Has to be called after each parameter update."""
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "post_parameter_update"):
                module.post_parameter_update()

    # docstr-coverage: inherited
    def collect_regularization_term(self):  # noqa: D102
        return sum(
            regularizer.pop_regularization_term()
            for regularizer in self.modules()
            if isinstance(regularizer, Regularizer)
        )


def _prepare_representation_module_list(
    max_id: int,
    shapes: Sequence[str],
    label: str,
    representations: OneOrManyHintOrType[Representation] = None,
    representation_kwargs: OneOrManyOptionalKwargs = None,
    skip_checks: bool = False,
) -> Sequence[Representation]:
    """
    Normalize list of representations and wrap into nn.ModuleList.

    .. note ::
        Important: use ModuleList to ensure that Pytorch correctly handles their devices and parameters

    :param representations:
        the representations, or hints for them.
    :param representation_kwargs:
        additional keyword-based parameters for instantiating representations from hints.
    :param max_id:
        the maximum representation ID. Newly instantiated representations will contain that many representations, and
        pre-instantiated ones have to provide at least that many.
    :param shapes:
        the symbolic shapes, which are used for shape verification, if skip_checks is False.
    :param label:
        a label to use for error messages (typically, "entities" or "relations").
    :param skip_checks:
        whether to skip shape verification.

    :return:
        a module list of instantiated representation modules.

    :raises ValueError:
        if the maximum ID or shapes do not match
    """
    # TODO: allow max_id being present in representation_kwargs; if it matches max_id
    # TODO: we could infer some shapes from the given interaction shape information
    rs = representation_resolver.make_many(representations, kwargs=representation_kwargs, max_id=max_id)

    # check max-id
    for r in rs:
        if r.max_id < max_id:
            raise ValueError(
                f"{r} only provides {r.max_id} {label} representations, but should provide {max_id}.",
            )
        elif r.max_id > max_id:
            logger.warning(
                f"{r} provides {r.max_id} {label} representations, although only {max_id} are needed."
                f"While this is not necessarily wrong, it can indicate an error where the number of {label} "
                f"representations was chosen wrong.",
            )

    rs = cast(Sequence[Representation], nn.ModuleList(rs))
    if skip_checks:
        return rs

    # check shapes
    if len(rs) != len(shapes):
        raise ValueError(
            f"Interaction function requires {len(shapes)} {label} representations, but {len(rs)} were given."
        )
    check_shapes(
        *zip(
            (r.shape for r in rs),
            shapes,
        ),
        raise_on_errors=True,
    )
    return rs


def repeat_if_necessary(
    scores: torch.FloatTensor,
    representations: Sequence[Representation],
    num: Optional[int],
) -> torch.FloatTensor:
    """
    Repeat score tensor if necessary.

    If a model does not have entity/relation representations, the scores for
    `score_{h,t}` / `score_r` are always the same. For efficiency, they are thus
    only computed once, but to meet the API, they have to be brought into the correct shape afterwards.

    :param scores: shape: (batch_size, ?)
        the score tensor
    :param representations:
        the representations. If empty (i.e. no representations for this 1:n scoring), repetition needs to be applied
    :param num:
        the number of times to repeat, if necessary.

    :return:
        the score tensor, which has been repeated, if necessary
    """
    if representations:
        return scores
    return scores.repeat(1, num)


class ERModel(
    Generic[HeadRepresentation, RelationRepresentation, TailRepresentation],
    _NewAbstractModel,
):
    """A commonly useful base for KGEMs using embeddings and interaction modules.

    This model does not use post-init hooks to automatically initialize all of its
    parameters. Rather, the call to :func:`Model.reset_parameters_` happens at the end of
    ``ERModel.__init__``. This is possible because all trainable parameters should necessarily
    be passed through the ``super().__init__()`` in subclasses of :class:`ERModel`.

    Other code can still be put after the call to ``super().__init__()`` in subclasses, such as
    registering regularizers (as done in :class:`pykeen.models.ConvKB` and :class:`pykeen.models.TransH`).
    ---
    citation:
        author: Ali
        year: 2021
        link: https://jmlr.org/papers/v22/20-825.html
        github: pykeen/pykeen
    """

    #: The entity representations
    entity_representations: Sequence[Representation]

    #: The relation representations
    relation_representations: Sequence[Representation]

    #: The weight regularizers
    weight_regularizers: List[Regularizer]

    #: The interaction function
    interaction: Interaction

    def __init__(
        self,
        *,
        triples_factory: KGInfo,
        interaction: Union[
            str,
            Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation],
            Type[Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation]],
        ],
        interaction_kwargs: OptionalKwargs = None,
        entity_representations: OneOrManyHintOrType[Representation] = None,
        entity_representations_kwargs: OneOrManyOptionalKwargs = None,
        relation_representations: OneOrManyHintOrType[Representation] = None,
        relation_representations_kwargs: OneOrManyOptionalKwargs = None,
        skip_checks: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the module.

        :param triples_factory:
            The triples factory facilitates access to the dataset.
        :param interaction: The interaction module (e.g., TransE)
        :param interaction_kwargs:
            Additional key-word based parameters given to the interaction module's constructor, if not already
            instantiated.
        :param entity_representations: The entity representation or sequence of representations
        :param entity_representations_kwargs:
            additional keyword-based parameters for instantiation of entity representations
        :param relation_representations: The relation representation or sequence of representations
        :param relation_representations_kwargs:
            additional keyword-based parameters for instantiation of relation representations
        :param skip_checks:
            whether to skip entity representation checks.
        :param kwargs:
            Keyword arguments to pass to the base model
        """
        # TODO: support "broadcasting" representation regularizers?
        # e.g. re-use the same regularizer for everything; or
        # pass a dictionary with keys "entity"/"relation";
        # values are either a regularizer hint (=the same regularizer for all repr); or a sequence of appropriate length
        super().__init__(triples_factory=triples_factory, **kwargs)
        self.interaction = interaction_resolver.make(interaction, pos_kwargs=interaction_kwargs)
        self.entity_representations = _prepare_representation_module_list(
            representations=entity_representations,
            representation_kwargs=entity_representations_kwargs,
            max_id=triples_factory.num_entities,
            shapes=self.interaction.full_entity_shapes(),
            label="entity",
            skip_checks=skip_checks,
        )
        self.relation_representations = _prepare_representation_module_list(
            representations=relation_representations,
            representation_kwargs=relation_representations_kwargs,
            max_id=triples_factory.num_relations,
            shapes=self.interaction.relation_shape,
            label="relation",
            skip_checks=skip_checks,
        )
        # Comment: it is important that the regularizers are stored in a module list, in order to appear in
        # model.modules(). Thereby, we can collect them automatically.
        self.weight_regularizers = nn.ModuleList()
        # Explicitly call reset_parameters to trigger initialization
        self.reset_parameters_()

    def append_weight_regularizer(
        self,
        parameter: Union[str, nn.Parameter, Iterable[Union[str, nn.Parameter]]],
        regularizer: HintOrType[Regularizer],
        regularizer_kwargs: OptionalKwargs = None,
        default_regularizer: HintOrType[Regularizer] = None,
        default_regularizer_kwargs: OptionalKwargs = None,
    ) -> None:
        """
        Add a model weight to a regularizer's weight list, and register the regularizer with the model.

        :param parameter:
            The parameter, either as name, or as nn.Parameter object. A list of available parameter names is shown by
             `sorted(dict(self.named_parameters()).keys())`.
        :param regularizer:
            the regularizer or a hint thereof
        :param regularizer_kwargs:
            additional keyword-based parameters for the regularizer's instantiation
        :param default_regularizer:
            the default regularizer; if None, use :attr:`regularizer_default`
        :param default_regularizer_kwargs:
            the default regularizer kwargs; if None, use :attr:`regularizer_default_kwargs`

        :raises KeyError: If an invalid parameter name was given
        """
        # instantiate regularizer
        regularizer = regularizer_resolver.make(
            *normalize_with_default(
                choice=regularizer,
                kwargs=regularizer_kwargs,
                default=default_regularizer or self.regularizer_default,
                default_kwargs=default_regularizer_kwargs or self.regularizer_default_kwargs,
            )
        )

        # normalize input
        if isinstance(parameter, (str, nn.Parameter)):
            parameter = [parameter]
        weights: Mapping[str, nn.Parameter] = dict(self.named_parameters())
        for param in parameter:
            if isinstance(param, str):
                if param not in weights:
                    raise KeyError(f"Invalid parameter_name={parameter}. Available are: {sorted(weights.keys())}.")
                param: nn.Parameter = weights[param]  # type: ignore
            regularizer.add_parameter(parameter=param)
        self.weight_regularizers.append(regularizer)

    def forward(
        self,
        h_indices: torch.LongTensor,
        r_indices: torch.LongTensor,
        t_indices: torch.LongTensor,
        slice_size: Optional[int] = None,
        slice_dim: int = 0,
        *,
        mode: Optional[InductiveMode],
    ) -> torch.FloatTensor:
        """Forward pass.

        This method takes head, relation and tail indices and calculates the corresponding scores.
        It supports broadcasting.

        :param h_indices:
            The head indices.
        :param r_indices:
            The relation indices.
        :param t_indices:
            The tail indices.
        :param slice_size:
            The slice size.
        :param slice_dim:
            The dimension along which to slice
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.

        :return:
            The scores

        :raises NotImplementedError:
            if score repetition becomes necessary
        """
        if not self.entity_representations or not self.relation_representations:
            raise NotImplementedError("repeat scores not implemented for general case.")
        h, r, t = self._get_representations(h=h_indices, r=r_indices, t=t_indices, mode=mode)
        return self.interaction.score(h=h, r=r, t=t, slice_size=slice_size, slice_dim=slice_dim)

    def score_hrt(self, hrt_batch: torch.LongTensor, *, mode: Optional[InductiveMode] = None) -> torch.FloatTensor:
        """Forward pass.

        This method takes head, relation and tail of each triple and calculates the corresponding score.

        :param hrt_batch: shape: (batch_size, 3), dtype: long
            The indices of (head, relation, tail) triples.
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.

        :return: shape: (batch_size, 1), dtype: float
            The score for each triple.
        """
        # Note: slicing cannot be used here: the indices for score_hrt only have a batch
        # dimension, and slicing along this dimension is already considered by sub-batching.
        # Note: we do not delegate to the general method for performance reasons
        # Note: repetition is not necessary here
        h, r, t = self._get_representations(h=hrt_batch[:, 0], r=hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode)
        return self.interaction.score_hrt(h=h, r=r, t=t)

    def _check_slicing(self, slice_size: Optional[int]) -> None:
        """Raise an error, if slicing is requested, but the model does not support it."""
        if not slice_size:
            return
        if get_batchnorm_modules(self):  # if there are any, this is truthy
            raise ValueError("This model does not support slicing, since it has batch normalization layers.")

    def score_t(
        self, hr_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:
        """Forward pass using right side (tail) prediction.

        This method calculates the score for all possible tails for each (head, relation) pair.

        :param hr_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, relation) pairs.
        :param slice_size:
            The slice size.
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.

        :return: shape: (batch_size, num_entities), dtype: float
            For each h-r pair, the scores for all possible tails.
        """
        self._check_slicing(slice_size=slice_size)
        h, r, t = self._get_representations(h=hr_batch[:, 0], r=hr_batch[:, 1], t=None, mode=mode)
        return repeat_if_necessary(
            scores=self.interaction.score_t(h=h, r=r, all_entities=t, slice_size=slice_size),
            representations=self.entity_representations,
            num=self._get_entity_len(mode=mode),
        )

    def score_h(
        self, rt_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:
        """Forward pass using left side (head) prediction.

        This method calculates the score for all possible heads for each (relation, tail) pair.

        :param rt_batch: shape: (batch_size, 2), dtype: long
            The indices of (relation, tail) pairs.
        :param slice_size:
            The slice size.
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.

        :return: shape: (batch_size, num_entities), dtype: float
            For each r-t pair, the scores for all possible heads.
        """
        self._check_slicing(slice_size=slice_size)
        h, r, t = self._get_representations(h=None, r=rt_batch[:, 0], t=rt_batch[:, 1], mode=mode)
        return repeat_if_necessary(
            scores=self.interaction.score_h(all_entities=h, r=r, t=t, slice_size=slice_size),
            representations=self.entity_representations,
            num=self._get_entity_len(mode=mode),
        )

    def score_r(
        self, ht_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:
        """Forward pass using middle (relation) prediction.

        This method calculates the score for all possible relations for each (head, tail) pair.

        :param ht_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, tail) pairs.
        :param slice_size:
            The slice size.
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.

        :return: shape: (batch_size, num_relations), dtype: float
            For each h-t pair, the scores for all possible relations.
        """
        self._check_slicing(slice_size=slice_size)
        h, r, t = self._get_representations(h=ht_batch[:, 0], r=None, t=ht_batch[:, 1], mode=mode)
        return repeat_if_necessary(
            scores=self.interaction.score_r(h=h, all_relations=r, t=t, slice_size=slice_size),
            representations=self.relation_representations,
            num=self.num_relations,
        )

    def _get_entity_representations_from_inductive_mode(
        self, *, mode: Optional[InductiveMode]
    ) -> Sequence[Representation]:
        """
        Return the entity representations for the given inductive mode.

        :param mode:
            the inductive mode

        :raises ValueError:
            if the model does not support the given inductive mode, e.g.,
            because it is purely transductive

        :return:
            the entity representations for the given inductive mode
        """
        if mode is not None:
            raise ValueError(f"{self.__class__.__name__} does not support inductive mode: {mode}")
        return self.entity_representations

    def _get_entity_len(self, *, mode: Optional[InductiveMode]) -> Optional[int]:  # noqa:D105
        """
        Return the number of entities for the given inductive mode.

        :param mode:
            the inductive mode

        :raises NotImplementedError:
            if the model does not support the given inductive mode, e.g.,
            because it is purely transductive

        :return:
            the number of entities in the given inductive mode
        """
        if mode is not None:
            raise NotImplementedError
        return self.num_entities

    def _get_representations(
        self,
        h: Optional[torch.LongTensor],
        r: Optional[torch.LongTensor],
        t: Optional[torch.LongTensor],
        *,
        mode: Optional[InductiveMode],
    ) -> Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation]:
        """Get representations for head, relation and tails."""
        head_representations = tail_representations = self._get_entity_representations_from_inductive_mode(mode=mode)
        head_representations = [head_representations[i] for i in self.interaction.head_indices()]
        tail_representations = [tail_representations[i] for i in self.interaction.tail_indices()]
        hr, rr, tr = [
            [representation(indices=indices) for representation in representations]
            for indices, representations in (
                (h, head_representations),
                (r, self.relation_representations),
                (t, tail_representations),
            )
        ]
        # normalization
        return cast(
            Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation],
            tuple(x[0] if len(x) == 1 else x for x in (hr, rr, tr)),
        )
