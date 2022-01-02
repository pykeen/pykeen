# -*- coding: utf-8 -*-

"""Analyze uncertainty.

.. todo::

    @mberr please give a bit of narrative on why you would want to use the uncertainty predictions,
    how to interpret the results (e.g., what's good/bad, what actions could be taken to make improvements),
    relevant citations, etc.
"""

from typing import Callable, NamedTuple, Optional

import torch

from .base import Model
from ..utils import get_dropout_modules

__all__ = [
    "MissingDropoutError",
    "UncertainPrediction",
    "predict_hrt_uncertain",
    "predict_h_uncertain",
    "predict_r_uncertain",
    "predict_t_uncertain",
    "predict_uncertain_helper",
]


class MissingDropoutError(ValueError):
    """Raised during uncertainty analysis if no dropout modules are present."""


class UncertainPrediction(NamedTuple):
    """A pair of predicted scores and corresponding uncertainty."""

    #: The scores
    score: torch.FloatTensor

    #: The uncertainty, in the same shape as scores
    uncertainty: torch.FloatTensor


@torch.inference_mode()
def predict_uncertain_helper(
    model: Model,
    batch: torch.LongTensor,
    score_method: Callable[..., torch.FloatTensor],
    num_samples: int,
    slice_size: Optional[int] = None,
) -> UncertainPrediction:
    """
    Predict with uncertainty estimates via Monte-Carlo dropout.

    .. note::
        the model will be set to evaluation mode, and all dropout layers will be set to training mode

    :param model:
        the model used for predicting scores
    :param batch:
        the batch on which to predict. Its shape and content has to match what
        the `score_method` requires.
    :param score_method:
        the base score method to use (from `score_{hrt,h,r,t}`)
    :param num_samples: > 1
        The number of samples to use. More samples lead to better estimates, but
        increase memory requirements and runtime.
    :param slice_size: >0
        The divisor for the scoring function when using slicing.

    :return:
        A tuple (score_mean, score_std) of the mean and std of the scores sampled
        from the dropout distribution. The std may be interpreted as a measure of
        uncertainty.

    :raises MissingDropoutError:
        if the model does not contain dropout layers.
    """
    dropout_modules = get_dropout_modules(model)
    if not dropout_modules:
        raise MissingDropoutError(
            "Model needs to contain at least one dropout layer to use the Monte-Carlo Dropout technique.",
        )

    # Enforce evaluation mode
    model.eval()

    # set dropout layers to training mode
    for module in dropout_modules:
        module.train()

    kwargs = dict()
    if slice_size is not None:
        kwargs["slice_size"] = slice_size

    # draw samples
    batch = batch.to(model.device)
    scores = torch.stack([score_method(batch, **kwargs) for _ in range(num_samples)], dim=0)
    if model.predict_with_sigmoid:
        scores = torch.sigmoid(scores)

    # compute mean and std
    return UncertainPrediction(score=scores.mean(dim=0), uncertainty=scores.std(dim=0))


def predict_hrt_uncertain(
    model: Model,
    hrt_batch: torch.LongTensor,
    num_samples: int = 5,
) -> UncertainPrediction:
    """
    Calculate the scores with uncertainty quantification via Monto-Carlo dropout as proposed in [berrendorf2021]_.

    :param model:
        the model used for predicting scores
    :param hrt_batch: shape: (number of triples, 3)
        The indices of (head, relation, tail) triples.
    :param num_samples: >1
        the number of samples to draw

    :return: shape: (number of triples, 1)
        The score for each triple, and an uncertainty score, where larger scores
        correspond to less certain predictions.

    .. seealso::
        :func:`pykeen.models.Model.score_hrt`, :func:`predict_uncertain_helper`

    Example Usage::

        from pykeen.pipeline import pipeline
        from pykeen.models.predict import predict_hrt_uncertain

        result = pipeline(dataset="nations", model="ERMLPE")
        prediction_with_uncertainty = predict_hrt_uncertain(
            model=result.model,
            hrt_batch=result.training.mapped_triples[0:8],
        )
    """
    return predict_uncertain_helper(
        model=model,
        batch=hrt_batch,
        score_method=model.score_hrt,
        num_samples=num_samples,
    )


def predict_h_uncertain(
    model: Model,
    rt_batch: torch.LongTensor,
    num_samples: int = 5,
    slice_size: Optional[int] = None,
) -> UncertainPrediction:
    """Forward pass using left side (head) prediction for obtaining scores of all possible heads.

    This method calculates the score for all possible heads for each (relation, tail)
    pair, as well as an uncertainty quantification.

    Additionally, the model is set to evaluation mode.

    .. note::

        If the model has been trained with inverse relations, the task of predicting
        the head entities becomes the task of predicting the tail entities of the
        inverse triples, i.e., $f(*,r,t)$ is predicted by means of $f(t,r_{inv},*)$.

    :param model:
        the model used for predicting scores
    :param rt_batch: shape: (batch_size, 2), dtype: long
        The indices of (relation, tail) pairs.
    :param slice_size: >0
        The divisor for the scoring function when using slicing.
    :param num_samples: >1
        the number of samples to draw

    :return: shape: (batch_size, num_entities), dtype: float
        For each r-t pair, the scores for all possible heads.
    """
    return predict_uncertain_helper(
        model=model,
        batch=rt_batch,
        score_method=model.score_h_inverse if model.use_inverse_triples else model.score_h,
        num_samples=num_samples,
        slice_size=slice_size,
    )


def predict_r_uncertain(
    model: Model,
    ht_batch: torch.LongTensor,
    num_samples: int = 5,
    slice_size: Optional[int] = None,
) -> UncertainPrediction:
    """Forward pass using middle (relation) prediction for obtaining scores of all possible relations.

    This method calculates the score for all possible relations for each (head, tail)
    pair, as well as an uncertainty quantification.

    Additionally, the model is set to evaluation mode.

    :param model:
        the model used for predicting scores
    :param ht_batch: shape: (batch_size, 2), dtype: long
        The indices of (head, tail) pairs.
    :param slice_size: >0
        The divisor for the scoring function when using slicing.
    :param num_samples: >1
        the number of samples to draw

    :return: shape: (batch_size, num_relations), dtype: float
        For each h-t pair, the scores for all possible relations.
    """
    return predict_uncertain_helper(
        model=model,
        batch=ht_batch,
        score_method=model.score_r,
        num_samples=num_samples,
        slice_size=slice_size,
    )


def predict_t_uncertain(
    model: Model,
    hr_batch: torch.LongTensor,
    num_samples: int = 5,
    slice_size: Optional[int] = None,
) -> UncertainPrediction:
    """Forward pass using right side (tail) prediction for obtaining scores of all possible tails.

    This method calculates the score for all possible tails for each (head, relation)
    pair, as well as an uncertainty quantification.

    Additionally, the model is set to evaluation mode.

    .. note::

        We only expect the right side-side predictions, i.e., $(h,r,*)$ to change its
        default behavior when the model has been trained with inverse relations
        (mainly because of the behavior of the LCWA training approach). This is why
        the :func:`predict_scores_all_heads` has different behavior depending on
        if inverse triples were used in training, and why this function has the same
        behavior regardless of the use of inverse triples.

    :param model:
        the model used for predicting scores
    :param hr_batch: shape: (batch_size, 2), dtype: long
        The indices of (head, relation) pairs.
    :param slice_size: >0
        The divisor for the scoring function when using slicing.
    :param num_samples: >1
        the number of samples to draw

    :return: shape: (batch_size, num_entities), dtype: float
        For each h-r pair, the scores for all possible tails.
    """
    return predict_uncertain_helper(
        model=model,
        batch=hr_batch,
        score_method=model.score_t,
        num_samples=num_samples,
        slice_size=slice_size,
    )
