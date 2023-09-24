# -*- coding: utf-8 -*-

"""
Analyze uncertainty.

Currently, all implemented approaches are based on Monte-Carlo dropout [gal2016]_.
Monte-Carlo dropout relies on the model having dropout layers. While dropout usually is
turned off for inference / evaluation mode, MC dropout leaves dropout enabled. Thereby,
if we run the same prediction method $k$ times, we get $k$ different predictions.
The variance of these predictions can be used as an approximation of uncertainty, where
larger variance indicates higher uncertainty in the predicted score.

The absolute variance is usually hard to interpret, but comparing the variances with each
other can help to identify which scores are more uncertain than others.

The following code-block sketches an example use case, where we train a model with a
classification loss, i.e., on the triple classification task.

.. code-block:: python

    from pykeen.pipeline import pipeline
    from pykeen.models.uncertainty import predict_hrt_uncertain

    # train model
    # note: as this is an example, the model is only trained for a few epochs,
    #       but not until convergence. In practice, you would usually first verify that
    #       the model is sufficiently good in prediction, before looking at uncertainty scores
    result = pipeline(dataset="nations", model="ERMLPE", loss="bcewithlogits")

    # predict triple scores with uncertainty
    prediction_with_uncertainty = predict_hrt_uncertain(
        model=result.model,
        hrt_batch=result.training.mapped_triples[0:8],
    )

    # use a larger number of samples, to increase quality of uncertainty estimate
    prediction_with_uncertainty = predict_hrt_uncertain(
        model=result.model,
        hrt_batch=result.training.mapped_triples[0:8],
        num_samples=100,
    )

    # get most and least uncertain prediction on training set
    prediction_with_uncertainty = predict_hrt_uncertain(
        model=result.model,
        hrt_batch=result.training.mapped_triples,
        num_samples=100,
    )
    df = result.training.tensor_to_df(
        result.training.mapped_triples,
        logits=prediction_with_uncertainty.score[:, 0],
        probability=prediction_with_uncertainty.score[:, 0].sigmoid(),
        uncertainty=prediction_with_uncertainty.uncertainty[:, 0],
    )
    print(df.nlargest(5, columns="uncertainty"))
    print(df.nsmallest(5, columns="uncertainty"))

A collection of related work on uncertainty quantification can be found here:
https://github.com/uncertainty-toolbox/uncertainty-toolbox/blob/master/docs/paper_list.md
"""

from typing import Callable, NamedTuple, Optional

import torch

from .base import Model
from ..typing import InductiveMode
from ..utils import get_dropout_modules

__all__ = [
    "predict_hrt_uncertain",
    "predict_h_uncertain",
    "predict_t_uncertain",
    "predict_r_uncertain",
    "predict_uncertain_helper",
    "MissingDropoutError",
    "UncertainPrediction",
]


class MissingDropoutError(ValueError):
    """Raised during uncertainty analysis if no dropout modules are present."""


class UncertainPrediction(NamedTuple):
    """
    A pair of predicted scores and corresponding uncertainty.

    Since the uncertainty scores come from Monte-Carlo dropout, they are guaranteed to be non-negative
    with larger scores indicating higher uncertainty.
    """

    #: The scores
    score: torch.FloatTensor

    #: The uncertainty, in the same shape as scores
    uncertainty: torch.FloatTensor

    @classmethod
    def from_scores(cls, scores: torch.Tensor):
        """Make an instance from scores."""
        return cls(score=scores.mean(dim=0), uncertainty=scores.var(dim=0))


@torch.inference_mode()
def predict_uncertain_helper(
    model: Model,
    batch: torch.LongTensor,
    score_method: Callable[..., torch.FloatTensor],
    num_samples: int,
    slice_size: Optional[int] = None,
    *,
    mode: Optional[InductiveMode],
) -> UncertainPrediction:
    """
    Predict with uncertainty estimates via Monte-Carlo dropout.

    :param model:
        the model used for predicting scores
    :param batch:
        the batch on which to predict. Its shape and content has to match what
        the `score_method` requires.
    :param score_method:
        the base score method to use (from `score_{hrt,h,r,t}`)
    :param num_samples: >1
        The number of samples to use. More samples lead to better estimates, but
        increase memory requirements and runtime.
    :param slice_size: >0
        The divisor for the scoring function when using slicing.
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.

    :return:
        A tuple (score_mean, score_std) of the mean and std of the scores sampled
        from the dropout distribution. The std may be interpreted as a measure of
        uncertainty.

    :raises MissingDropoutError:
        if the model does not contain dropout layers.

    .. warning::
        This function sets the model to evaluation mode and all dropout layers to
        training mode.
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

    # draw samples
    batch = batch.to(model.device)
    kwargs = {}
    if slice_size is not None:
        kwargs["slice_size"] = slice_size
    scores = torch.stack([score_method(batch, mode=mode, **kwargs) for _ in range(num_samples)], dim=0)
    if model.predict_with_sigmoid:
        scores = torch.sigmoid(scores)

    # compute mean and std
    return UncertainPrediction.from_scores(scores)


def predict_hrt_uncertain(
    model: Model,
    hrt_batch: torch.LongTensor,
    num_samples: int = 5,
    *,
    mode: Optional[InductiveMode] = None,
) -> UncertainPrediction:
    """
    Calculate the scores with uncertainty quantification via Monte-Carlo dropout.

    :param model:
        the model used for predicting scores
    :param hrt_batch: shape: (number of triples, 3)
        The indices of (head, relation, tail) triples.
    :param num_samples: >1
        the number of samples to draw
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.

    :return: shape: (number of triples, 1)
        The score for each triple, and an uncertainty score, where larger scores
        correspond to less certain predictions.

        This function delegates to :func:`predict_uncertain_helper` by using
        :func:`pykeen.models.Model.score_hrt` as the ``score_method``.

    .. warning::
        This function sets the model to evaluation mode and all dropout layers
        to training mode.

    Example Usage::

        from pykeen.pipeline import pipeline
        from pykeen.models.uncertainty import predict_hrt_uncertain

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
        mode=mode,
    )


def predict_h_uncertain(
    model: Model,
    rt_batch: torch.LongTensor,
    num_samples: int = 5,
    slice_size: Optional[int] = None,
    *,
    mode: Optional[InductiveMode] = None,
) -> UncertainPrediction:
    """Forward pass using left side (head) prediction for obtaining scores of all possible heads.

    This method calculates the score for all possible heads for each (relation, tail)
    pair, as well as an uncertainty quantification.

    .. note::

        If the model has been trained with inverse relations, the task of predicting
        the head entities becomes the task of predicting the tail entities of the
        inverse triples, i.e., $f(*,r,t)$ is predicted by means of $f(t,r_{inv},*)$.

    :param model:
        the model used for predicting scores
    :param rt_batch: shape: (batch_size, 2)
        The indices of (relation, tail) pairs.
    :param slice_size: >0
        The divisor for the scoring function when using slicing.
    :param num_samples: >1
        the number of samples to draw
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.

    :return: shape: (batch_size, num_entities)
        For each r-t pair, the scores for all possible heads.

        This function delegates to :func:`predict_uncertain_helper` by using
        :func:`pykeen.models.Model.score_h` (or :func:`pykeen.models.Model.score_h_inverse`
        if the model uses inverse triples) as the ``score_method``.

    .. warning::
        This function sets the model to evaluation mode and all dropout layers
        to training mode.
    """
    return predict_uncertain_helper(
        model=model,
        batch=rt_batch,
        score_method=model.score_h_inverse if model.use_inverse_triples else model.score_h,
        num_samples=num_samples,
        slice_size=slice_size,
        mode=mode,
    )


def predict_r_uncertain(
    model: Model,
    ht_batch: torch.LongTensor,
    num_samples: int = 5,
    slice_size: Optional[int] = None,
    *,
    mode: Optional[InductiveMode] = None,
) -> UncertainPrediction:
    """Forward pass using middle (relation) prediction for obtaining scores of all possible relations.

    This method calculates the score for all possible relations for each (head, tail)
    pair, as well as an uncertainty quantification.

    :param model:
        the model used for predicting scores
    :param ht_batch: shape: (batch_size, 2)
        The indices of (head, tail) pairs.
    :param slice_size: >0
        The divisor for the scoring function when using slicing.
    :param num_samples: >1
        the number of samples to draw
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.

    :return: shape: (batch_size, num_relations)
        For each h-t pair, the scores for all possible relations.

        This function delegates to :func:`predict_uncertain_helper` by using
        :func:`pykeen.models.Model.score_r` as the ``score_method``.

    .. warning::
        This function sets the model to evaluation mode and all dropout layers
        to training mode.
    """
    return predict_uncertain_helper(
        model=model,
        batch=ht_batch,
        score_method=model.score_r,
        num_samples=num_samples,
        slice_size=slice_size,
        mode=mode,
    )


def predict_t_uncertain(
    model: Model,
    hr_batch: torch.LongTensor,
    num_samples: int = 5,
    slice_size: Optional[int] = None,
    *,
    mode: Optional[InductiveMode] = None,
) -> UncertainPrediction:
    """Forward pass using right side (tail) prediction for obtaining scores of all possible tails.

    This method calculates the score for all possible tails for each (head, relation)
    pair, as well as an uncertainty quantification.

    .. note::

        We only expect the right side predictions, i.e., $(h,r,*)$ to change its
        default behavior when the model has been trained with inverse relations
        (mainly because of the behavior of the LCWA training approach). This is why
        the :func:`predict_h_uncertain` has different
        behavior depending on if inverse triples were used in training, and why
        this function has the same behavior regardless of the use of inverse triples.

    :param model:
        the model used for predicting scores
    :param hr_batch: shape: (batch_size, 2)
        The indices of (head, relation) pairs.
    :param slice_size: >0
        The divisor for the scoring function when using slicing.
    :param num_samples: >1
        the number of samples to draw
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.

    :return: shape: (batch_size, num_entities)
        For each h-r pair, the scores for all possible tails.

        This function delegates to :func:`predict_uncertain_helper` by using
        :func:`pykeen.models.Model.score_t` as the ``score_method``.

    .. warning::
        This function sets the model to evaluation mode and all dropout layers
        to training mode.
    """
    return predict_uncertain_helper(
        model=model,
        batch=hr_batch,
        score_method=model.score_t,
        num_samples=num_samples,
        slice_size=slice_size,
        mode=mode,
    )
