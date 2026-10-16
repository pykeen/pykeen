"""Base module for all KGE models."""

from __future__ import annotations

import inspect
import logging
import os
import pickle
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, ClassVar, TypedDict, Unpack, final

import torch
from class_resolver import HintOrType
from docdata import parse_docdata
from torch import nn

from .scoring import ScoringBatch, TargetScoringBatch, TripleScoringBatch
from ..constants import TARGET_TO_KEYS
from ..inverse import RelationInverter, relation_inverter_resolver
from ..losses import Loss, MarginRankingLoss, loss_resolver
from ..triples import KGInfo
from ..typing import (
    LABEL_HEAD,
    LABEL_RELATION,
    LABEL_TAIL,
    FloatTensor,
    InductiveMode,
    LongTensor,
    MappedTriples,
    Target,
)
from ..utils import (
    NoRandomSeedNecessary,
    get_batchnorm_modules,
    get_dropout_modules,
    get_preferred_device,
    set_random_seed,
)

__all__ = [
    "Model",
    "PredictKwargs",
]

logger = logging.getLogger(__name__)


class PredictKwargs(TypedDict, total=False):
    """Keyword-based parameters shared by the prediction methods of :class:`Model`."""

    #: the maximum number of candidates to score at once; only supported for 1:n scoring. Defaults to no slicing.
    slice_size: int | None

    #: whether to score by means of the inverse triples, which requires the model to use inverse relations.
    #: If None, the head predictions are inverted iff the model uses inverse relations, and all others are not.
    invert: bool | None

    #: the pass mode, which is None in the transductive setting and one of "training", "validation", or "testing"
    #: in the inductive setting. Defaults to None.
    mode: InductiveMode | None

    #: whether to enable dropout during prediction (Monte-Carlo dropout); the dropout modules are restored afterwards.
    #: Defaults to False.
    mc_dropout: bool


@contextmanager
def _monte_carlo_dropout(model: nn.Module, enabled: bool) -> Iterator[None]:
    """Temporarily put all dropout modules of a model into training mode, if enabled."""
    if not enabled:
        yield
        return
    modules = get_dropout_modules(model)
    previous = [module.training for module in modules]
    for module in modules:
        module.train()
    try:
        yield
    finally:
        for module, training in zip(modules, previous, strict=True):
            module.train(training)


def iter_slices(
    ids: LongTensor | None, slice_size: int, total: int, device: torch.device, dim: int = -1
) -> Iterator[LongTensor]:
    """
    Iterate over slices of an (implicit) index tensor.

    :param ids:
        The ID tensor, or None to indicate :code:`torch.arange(total)`
    :param slice_size: $>0$
        The (maximum) size of each individual slice.
    :param total:
        The total number of IDs; only used when ``ids`` is ``None``.
    :param device:
        The device on which to create the slices of the implicit :code:`torch.arange(total)`.
    :param dim:
        The dimension along which to slice.

    :yields: Slices of the index tensor
    """
    if ids is None:
        for start in range(0, total, slice_size):
            yield torch.arange(start=start, end=min(start + slice_size, total), device=device)
    else:
        yield from ids.split(split_size=slice_size, dim=dim)


class Model(nn.Module, ABC):
    """A base module for KGE models.

    Subclasses of :class:`Model` can decide however they want on how to store entities' and
    relations' representations, how they want to be looked up, and how they should
    be scored. The :class:`~pykeen.models.ERModel` provides a commonly used interface for models storing entity
    and relation representations in the form of :class:`~pykeen.nn.representation.Embedding`.

    .. note::

        The scoring interface is organized in layers, which use different relation ID spaces.

        The ``predict_*`` methods, e.g., :meth:`Model.predict_hrt`, are the user-facing interface for inference. They
        expect the *real* relation IDs, i.e., the ones stored in the triples factory's
        :attr:`~pykeen.triples.TriplesFactory.relation_to_id` and ``mapped_triples``, accept tensors
        on any device, and additionally take care of setting the evaluation mode, of
        optionally applying a sigmoid, and of using the inverse triples for head prediction. Their shared keyword-based
        parameters are described by :class:`PredictKwargs`.

        :meth:`Model.forward` is the entry point for training. It receives a
        :class:`~pykeen.models.scoring.ScoringBatch` with the model's *internal* relation IDs, which comprise a forward
        and an inverse ID for each relation when the model is trained with inverse relations,
        cf. :class:`~pykeen.inverse.RelationInverter`, and which has to be on the model's device. It takes care of
        slicing. The training triples with internal relation IDs are obtained via
        :meth:`Model.prepare_training_triples`.

        Subclasses only implement :meth:`Model._score`, which does not need to care about any of this.

        Unless the model uses inverse relations, the two ID spaces coincide.
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]]

    _random_seed: int | None

    #: The default loss function class
    loss_default: ClassVar[type[Loss]] = MarginRankingLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any] | None] = {"margin": 1.0, "reduction": "mean"}
    #: The instance of the loss
    loss: Loss

    #: the number of entities
    num_entities: int
    #: the number of relations
    num_relations: int
    #: whether to use inverse relations
    use_inverse_triples: bool
    #: utility for generating inverse relations
    relation_inverter: RelationInverter

    #: When predict_with_sigmoid is set to True, the sigmoid function is
    #: applied to the logits during evaluation and also for predictions
    #: after training, but has no effect on the training.
    predict_with_sigmoid: bool

    def __init__(
        self,
        *,
        triples_factory: KGInfo,
        loss: HintOrType[Loss] = None,
        loss_kwargs: Mapping[str, Any] | None = None,
        predict_with_sigmoid: bool = False,
        random_seed: int | None = None,
        use_inverse_triples: bool = False,
        relation_inverter: HintOrType[RelationInverter] = None,
    ) -> None:
        """Initialize the module.

        :param triples_factory:
            The triples factory facilitates access to the dataset.
        :param loss:
            The loss to use. If None is given, use the loss default specific to the model subclass.
        :param loss_kwargs:
            keyword-based parameters passed to the loss instance upon instantiation
        :param predict_with_sigmoid:
            Whether to apply sigmoid onto the scores when predicting scores. Applying sigmoid at prediction time may
            lead to exactly equal scores for certain triples with very high, or very low score. When not trained with
            applying sigmoid (or using BCEWithLogitsLoss), the scores are not calibrated to perform well with sigmoid.
        :param random_seed:
            A random seed to use for initialising the model's weights. **Should** be set when aiming at reproducibility.
        :param use_inverse_triples:
            Whether to model inverse relations, i.e., to add an artificial inverse relation for each relation, and
            train on the inverse triples, too.
        :param relation_inverter:
            A relation inverter or a hint for constructing one
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

        self.use_inverse_triples = use_inverse_triples
        self.num_entities = triples_factory.num_entities
        self.num_relations = (
            2 * triples_factory.real_num_relations if use_inverse_triples else triples_factory.real_num_relations
        )
        self.relation_inverter = relation_inverter_resolver.make(relation_inverter)

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

        :param kwargs:
            ignored keyword-based parameters
        """
        if not inspect.isabstract(cls):
            parse_docdata(cls)

    @property
    def device(self) -> torch.device:
        """Return the model's device."""
        return get_preferred_device(self, allow_ambiguity=False)

    def reset_parameters_(self):
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
    def _reset_parameters_(self):
        """Reset all parameters of the model in-place."""

    @abstractmethod
    def _get_entity_len(self, *, mode: InductiveMode | None) -> int:
        """Get the number of entities depending on the mode parameters."""

    def post_parameter_update(self) -> None:
        """Has to be called after each parameter update."""

    """Scoring methods"""

    @final
    def forward(  # type: ignore[override]
        self,
        batch: ScoringBatch,
        *,
        slice_size: int | None = None,
        mode: InductiveMode | None = None,
    ) -> FloatTensor:
        """Calculate the scores for a scoring request.

        This is the entry point for anything which should be differentiable, e.g., training. It operates on the
        model's *internal* relation IDs, and expects the batch to be on the model's device; use the ``predict_*``
        methods for inference on *real* relation IDs.

        :param batch:
            the scoring request, with internal relation IDs
        :param slice_size: >0
            the maximum number of candidates to score at once; only supported for 1:n scoring
        :param mode:
            the pass mode, which is None in the transductive setting and one of "training", "validation", or
            "testing" in the inductive setting

        :raises ValueError:
            if slicing is requested for a batch without a scoring target

        :return: shape: (*batch_shape,) or (*batch_shape, num)
            the scores
        """
        match batch:
            case TripleScoringBatch():
                if slice_size:
                    raise ValueError("Slicing requires a target; there is nothing to slice along.")
                return self._score(batch, mode=mode)
            case TargetScoringBatch():
                total = self.num_relations if batch.target == LABEL_RELATION else self._get_entity_len(mode=mode)
                # normalize before checking
                if slice_size and slice_size >= total:
                    slice_size = None
                self._check_slicing(slice_size=slice_size)
                if not slice_size:
                    return self._score(batch, mode=mode)
                # slice early to allow lazy computation of target representations
                return torch.cat(
                    [
                        self._score(batch.with_target_ids(partial_ids), mode=mode)
                        for partial_ids in iter_slices(
                            ids=batch.target_ids, slice_size=slice_size, total=total, device=batch.device
                        )
                    ],
                    dim=-1,
                )
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    @abstractmethod
    def _score(self, batch: ScoringBatch, *, mode: InductiveMode | None = None) -> FloatTensor:
        """Calculate the scores for a scoring request.

        This is the single method to implement for scoring. It receives the model's *internal* relation IDs on the
        model's device, and does not have to care about slicing, inverse relations, evaluation mode, or sigmoid.

        :param batch:
            the scoring request, with internal relation IDs
        :param mode:
            the pass mode, which is None in the transductive setting and one of "training", "validation", or
            "testing" in the inductive setting

        :return: shape: (*batch_shape,) or (*batch_shape, num)
            the scores
        """

    def _check_slicing(self, slice_size: int | None) -> None:
        """Raise an error, if slicing is requested, but the model does not support it."""
        if not slice_size:
            return
        # batch normalization modules use batch statistics in training mode
        # -> different batch divisions lead to different results
        if self.training and get_batchnorm_modules(self):
            raise ValueError("This model does not support slicing, since it has batch normalization layers.")

    @abstractmethod
    def collect_regularization_term(self) -> FloatTensor:
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

    def save_state(self, path: str | os.PathLike) -> None:
        """Save the state of the model.

        :param path:
            Path of the file where to store the state in.
        """
        torch.save(self.state_dict(), path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, path: str | os.PathLike) -> None:
        """Load the state of the model.

        :param path:
            Path of the file where to load the state from.
        """
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))

    """Prediction methods"""

    @final
    def prepare_training_triples(self, mapped_triples: MappedTriples) -> MappedTriples:
        """Prepare triples for training the model, i.e., convert the real relation IDs to the internal ones.

        When the model is trained with inverse relations, the inverse of each triple is added, too, cf.
        :class:`~pykeen.inverse.RelationInverter`. Otherwise, the real and the internal relation IDs coincide, and the
        triples are returned unchanged.

        :param mapped_triples: shape: (n, 3)
            the ID-based triples, using real relation IDs

        :return: shape: (n, 3) or (2n, 3)
            the triples, with internal relation IDs and inverse triples, if the model uses inverse relations
        """
        if not self.use_inverse_triples:
            return mapped_triples

        logger.info("Creating inverse triples.")
        return torch.cat(
            [
                self.relation_inverter.to_internal_batch(batch=mapped_triples),
                self.relation_inverter.to_internal_batch(batch=mapped_triples, invert=True).flip(1),
            ]
        )

    @final
    def _predict_batch(
        self,
        batch: ScoringBatch,
        *,
        slice_size: int | None = None,
        invert: bool | None = None,
        mode: InductiveMode | None = None,
        mc_dropout: bool = False,
    ) -> FloatTensor:
        r"""Calculate the scores for a scoring request with real relation IDs, in evaluation mode.

        :param batch:
            the scoring request, on the model's device, with the *real* relation IDs
        :param slice_size:
            the maximum number of candidates to score at once, cf. :class:`PredictKwargs`
        :param invert:
            whether to score by means of the inverse triples, cf. :class:`PredictKwargs`. If None, the head
            predictions are made by means of the inverse triples, i.e., $f(*,r,t)$ is predicted via $f(t,r_{inv},*)$,
            if the model uses inverse relations.
        :param mode:
            the pass mode, cf. :class:`PredictKwargs`
        :param mc_dropout:
            whether to enable dropout, cf. :class:`PredictKwargs`

        :raises ValueError:
            if inversion is requested, but the model does not use inverse relations

        :return: shape: (*batch_shape,) or (*batch_shape, num)
            the scores, passed through a sigmoid if :attr:`predict_with_sigmoid` is set
        """
        if invert is None:
            invert = self.use_inverse_triples and isinstance(batch, TargetScoringBatch) and batch.target == LABEL_HEAD

        self.eval()  # Enforce evaluation mode
        # the two relation ID spaces only differ if the model uses inverse relations
        if self.use_inverse_triples:
            batch = batch.to_internal(self.relation_inverter)
        if invert:
            if not self.use_inverse_triples:
                raise ValueError(
                    "Your model is not configured to predict with inverse relations."
                    " Set ``use_inverse_triples=True`` when creating the model.",
                )
            batch = batch.invert(self.relation_inverter)

        with _monte_carlo_dropout(self, enabled=mc_dropout):
            scores = self(batch, slice_size=slice_size, mode=mode)
        if self.predict_with_sigmoid:
            scores = torch.sigmoid(scores)
        return scores

    def _to_device(self, tensor: LongTensor | None) -> LongTensor | None:
        """Move an optional tensor onto the model's device."""
        return None if tensor is None else tensor.to(self.device)

    @final
    def predict_hrt(self, hrt_batch: LongTensor, **kwargs: Unpack[PredictKwargs]) -> FloatTensor:
        """Calculate the scores for triples.

        This method takes head, relation and tail of each triple and calculates the corresponding score.

        Additionally, the model is set to evaluation mode.

        :param hrt_batch: shape: (number of triples, 3), dtype: long
            The indices of (head, relation, tail) triples.
        :param kwargs:
            additional keyword-based parameters, cf. :class:`PredictKwargs`; ``slice_size`` is not supported.

        :return: shape: (number of triples, 1), dtype: float
            The score for each triple.
        """
        batch = TripleScoringBatch.from_batch(hrt_batch.to(self.device))
        return self._predict_batch(batch, **kwargs).unsqueeze(dim=-1)

    @final
    def predict_h(
        self,
        rt_batch: LongTensor,
        *,
        heads: LongTensor | None = None,
        **kwargs: Unpack[PredictKwargs],
    ) -> FloatTensor:
        """Forward pass using left side (head) prediction for obtaining scores of all possible heads.

        This method calculates the score for all possible heads for each (relation, tail) pair.

        .. note::

            If the model has been trained with inverse relations, the task of predicting
            the head entities becomes the task of predicting the tail entities of the
            inverse triples, i.e., $f(*,r,t)$ is predicted by means of $f(t,r_{inv},*)$.
            This can be controlled via the ``invert`` parameter.

        Additionally, the model is set to evaluation mode.

        :param rt_batch: shape: (batch_size, 2), dtype: long
            The indices of (relation, tail) pairs.
        :param heads: shape: (num_heads,) | (batch_size, num_heads)
            head entity indices to score against. If None, scores against all entities (from the given mode).
        :param kwargs:
            additional keyword-based parameters, cf. :class:`PredictKwargs`

        :return: shape: (batch_size, num_heads), dtype: float
            For each r-t pair, the scores for all possible heads.
        """
        rt_batch = rt_batch.to(self.device)
        batch = TargetScoringBatch.from_transposed_batch(
            head=self._to_device(heads), relation=rt_batch[:, 0], tail=rt_batch[:, 1], target=LABEL_HEAD
        )
        return self._predict_batch(batch, **kwargs)

    @final
    def predict_t(
        self,
        hr_batch: LongTensor,
        *,
        tails: LongTensor | None = None,
        **kwargs: Unpack[PredictKwargs],
    ) -> FloatTensor:
        """Forward pass using right side (tail) prediction for obtaining scores of all possible tails.

        This method calculates the score for all possible tails for each (head, relation) pair.

        Additionally, the model is set to evaluation mode.

        :param hr_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, relation) pairs.
        :param tails: shape: (num_tails,) | (batch_size, num_tails)
            tail entity indices to score against. If None, scores against all entities (from the given mode).
        :param kwargs:
            additional keyword-based parameters, cf. :class:`PredictKwargs`

        :return: shape: (batch_size, num_tails), dtype: float
            For each h-r pair, the scores for all possible tails.

        .. note::

            We only expect the right side-predictions, i.e., $(h,r,*)$ to change its
            default behavior when the model has been trained with inverse relations
            (mainly because of the behavior of the LCWA training approach). This is why
            the :func:`predict_h` has different behavior depending on
            if inverse triples were used in training, and why this function has the same
            behavior regardless of the use of inverse triples.
        """
        hr_batch = hr_batch.to(self.device)
        batch = TargetScoringBatch.from_transposed_batch(
            head=hr_batch[:, 0], relation=hr_batch[:, 1], tail=self._to_device(tails), target=LABEL_TAIL
        )
        return self._predict_batch(batch, **kwargs)

    @final
    def predict_r(
        self,
        ht_batch: LongTensor,
        *,
        relations: LongTensor | None = None,
        **kwargs: Unpack[PredictKwargs],
    ) -> FloatTensor:
        """Forward pass using middle (relation) prediction for obtaining scores of all possible relations.

        This method calculates the score for all possible relations for each (head, tail) pair.

        Additionally, the model is set to evaluation mode.

        :param ht_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, tail) pairs.
        :param relations: shape: (num_real_relations,)
            The relations to score against, given as *real* relation IDs. If None, scores against
            all real relations.
        :param kwargs:
            additional keyword-based parameters, cf. :class:`PredictKwargs`

        :return: shape: (batch_size, num_real_relations), dtype: float
            For each h-t pair, the scores for all possible relations, with the columns indexed by
            *real* relation ID. The artificial inverse relations are never scored against.
        """
        ht_batch = ht_batch.to(self.device)
        relations = self._to_device(relations)
        if self.use_inverse_triples and relations is None:
            # the model's internal relations comprise the artificial inverse ones. Asking for the
            # forward IDs explicitly keeps the columns indexed by "real" relation ID, without
            # computing the inverse scores just to discard them.
            relations = torch.arange(self.num_real_relations, device=self.device)
        batch = TargetScoringBatch.from_transposed_batch(
            head=ht_batch[:, 0], relation=relations, tail=ht_batch[:, 1], target=LABEL_RELATION
        )
        return self._predict_batch(batch, **kwargs)

    @final
    def predict(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        full_batch: bool = True,
        ids: LongTensor | None = None,
        **kwargs: Unpack[PredictKwargs],
    ) -> FloatTensor:
        """
        Predict scores for the given target.

        :param hrt_batch: shape: (batch_size, 3) or (batch_size, 2)
            the full batch, or the relevant part of it
        :param target:
            the target to predict
        :param full_batch:
            whether `hrt_batch` is the full batch, or only the "input" part of the target prediction method
        :param ids:
            restrict prediction to only those ids
        :param kwargs:
            additional keyword-based parameters, cf. :class:`PredictKwargs`

        :raises ValueError:
            if the target is invalid

        :return: shape: (batch_size, num)
            the scores
        """
        if target == LABEL_TAIL:
            if full_batch:
                hrt_batch = hrt_batch[:, TARGET_TO_KEYS[LABEL_TAIL]]
            return self.predict_t(hrt_batch, **kwargs, tails=ids)

        if target == LABEL_RELATION:
            if full_batch:
                hrt_batch = hrt_batch[:, TARGET_TO_KEYS[LABEL_RELATION]]
            return self.predict_r(hrt_batch, **kwargs, relations=ids)

        if target == LABEL_HEAD:
            if full_batch:
                hrt_batch = hrt_batch[:, TARGET_TO_KEYS[LABEL_HEAD]]
            return self.predict_h(hrt_batch, **kwargs, heads=ids)

        raise ValueError(f"Unknown target={target}")
