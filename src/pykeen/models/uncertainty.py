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

from typing import NamedTuple

import torch

from .base import Model
from .scoring import ScoringBatch, TargetScoringBatch, TripleScoringBatch
from ..typing import LABEL_HEAD, LABEL_RELATION, LABEL_TAIL, FloatTensor, InductiveMode, LongTensor
from ..utils import get_dropout_modules

__all__ = [
    "MissingDropoutError",
    "UncertainPrediction",
    "predict_h_uncertain",
    "predict_hrt_uncertain",
    "predict_r_uncertain",
    "predict_t_uncertain",
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
    score: FloatTensor

    #: The uncertainty, in the same shape as scores
    uncertainty: FloatTensor

    @classmethod
    def from_scores(cls, scores: torch.Tensor):
        """Make an instance from scores."""
        return cls(score=scores.mean(dim=0), uncertainty=scores.var(dim=0))


@torch.inference_mode()
def _predict_uncertain_helper(
    model: Model,
    batch: ScoringBatch,
    num_samples: int,
    slice_size: int | None = None,
    *,
    mode: InductiveMode | None,
    invert: bool | None = None,
) -> UncertainPrediction:
    """
    Predict with uncertainty estimates via Monte-Carlo dropout.

    :param model:
        the model used for predicting scores
    :param batch:
        the scoring request, with the *real* relation IDs, on the model's device
    :param num_samples: >1
        The number of samples to use. More samples lead to better estimates, but
        increase memory requirements and runtime.
    :param slice_size: >0
        The divisor for the scoring function when using slicing.
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.
    :param invert:
        whether to score by means of the inverse triples, cf. :class:`~pykeen.models.base.PredictKwargs`

    :return:
        A tuple (score_mean, score_std) of the mean and std of the scores sampled
        from the dropout distribution. The std may be interpreted as a measure of
        uncertainty.

    :raises MissingDropoutError:
        if the model does not contain dropout layers.

    .. warning::
        This function sets the model to evaluation mode. Dropout layers are only enabled
        while drawing the samples.
    """
    if not get_dropout_modules(model):
        raise MissingDropoutError(
            "Model needs to contain at least one dropout layer to use the Monte-Carlo Dropout technique.",
        )

    # draw samples; note: this also applies the sigmoid, if configured
    scores = torch.stack(
        [
            model._predict_batch(batch, slice_size=slice_size, mode=mode, invert=invert, mc_dropout=True)
            for _ in range(num_samples)
        ],
        dim=0,
    )

    # compute mean and std
    return UncertainPrediction.from_scores(scores)


def predict_hrt_uncertain(
    model: Model,
    hrt_batch: LongTensor,
    num_samples: int = 5,
    *,
    mode: InductiveMode | None = None,
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

    .. warning::
        This function sets the model to evaluation mode. Dropout layers are only enabled
        while drawing the samples.

    Example Usage::

        from pykeen.pipeline import pipeline
        from pykeen.models.uncertainty import predict_hrt_uncertain

        result = pipeline(dataset="nations", model="ERMLPE")
        prediction_with_uncertainty = predict_hrt_uncertain(
            model=result.model,
            hrt_batch=result.training.mapped_triples[0:8],
        )
    """
    prediction = _predict_uncertain_helper(
        model=model,
        batch=TripleScoringBatch.from_batch(hrt_batch.to(model.device)),
        num_samples=num_samples,
        mode=mode,
    )
    # the scores of triples have shape (batch_size,); add the trailing dimension, as for the other prediction methods
    return UncertainPrediction(
        score=prediction.score.unsqueeze(dim=-1), uncertainty=prediction.uncertainty.unsqueeze(dim=-1)
    )


def predict_h_uncertain(
    model: Model,
    rt_batch: LongTensor,
    num_samples: int = 5,
    slice_size: int | None = None,
    *,
    mode: InductiveMode | None = None,
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

    .. warning::
        This function sets the model to evaluation mode. Dropout layers are only enabled
        while drawing the samples.
    """
    return _predict_uncertain_helper(
        model=model,
        batch=TargetScoringBatch.from_transposed_batch(
            head=None, relation=rt_batch[:, 0].to(model.device), tail=rt_batch[:, 1].to(model.device), target=LABEL_HEAD
        ),
        num_samples=num_samples,
        slice_size=slice_size,
        mode=mode,
    )


def predict_r_uncertain(
    model: Model,
    ht_batch: LongTensor,
    num_samples: int = 5,
    slice_size: int | None = None,
    *,
    mode: InductiveMode | None = None,
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

    .. warning::
        This function sets the model to evaluation mode. Dropout layers are only enabled
        while drawing the samples.
    """
    return _predict_uncertain_helper(
        model=model,
        batch=TargetScoringBatch.from_transposed_batch(
            head=ht_batch[:, 0].to(model.device),
            relation=torch.arange(model.num_real_relations, device=model.device),
            tail=ht_batch[:, 1].to(model.device),
            target=LABEL_RELATION,
        ),
        num_samples=num_samples,
        slice_size=slice_size,
        mode=mode,
    )


def predict_t_uncertain(
    model: Model,
    hr_batch: LongTensor,
    num_samples: int = 5,
    slice_size: int | None = None,
    *,
    mode: InductiveMode | None = None,
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

    .. warning::
        This function sets the model to evaluation mode. Dropout layers are only enabled
        while drawing the samples.
    """
    return _predict_uncertain_helper(
        model=model,
        batch=TargetScoringBatch.from_transposed_batch(
            head=hr_batch[:, 0].to(model.device), relation=hr_batch[:, 1].to(model.device), tail=None, target=LABEL_TAIL
        ),
        num_samples=num_samples,
        slice_size=slice_size,
        mode=mode,
    )
