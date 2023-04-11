# -*- coding: utf-8 -*-

"""
Prediction workflows.

.. _predictions:

After training, the interaction model (e.g., TransE, ConvE, RotatE) can assign a score to an arbitrary triple,
whether it appeared during training, testing, or not. In PyKEEN, each is implemented such that the higher the score
(or less negative the score), the more likely a triple is to be true.

However, for most models, these scores do not have obvious statistical interpretations. This has two main consequences:

1. The score for a triple from one model can not be compared to the score for that triple from another model
2. There is no *a priori* minimum score for a triple to be labeled as true, so predictions must be given as
   a prioritization by sorting a set of triples by their respective scores.

For the remainder of this part of the documentation, we assume that we have trained a model, e.g. via

>>> from pykeen.pipeline import pipeline
>>> result = pipeline(dataset="nations", model="pairre", training_kwargs=dict(num_epochs=0))


High-Level
==========
The prediction workflow offers three high-level methods to perform predictions

- :func:`pykeen.predict.predict_triples` can be used to calculate scores for a given set of triples.
- :func:`pykeen.predict.predict_target` can be used to score choices for a given prediction target, i.e.
  calculate scores for head entities, relations, or tail entities given the other two.
- :func:`pykeen.predict.predict_all` can be used to calculate scores for all possible triples.
  Scientifically, :func:`pykeen.predict.predict_all` is the most interesting in a scenario where
  predictions could be tested and validated experimentally.

.. warning ::
    Please note that not all models automatically have interpretable scores, and their calibration may be poor. Thus,
    exercise caution when interpreting the results.


Triple Scoring
--------------

When scoring triples with :func:`pykeen.predict.predict_triples`, we obtain a score for each of the given
triples. As an example, we will calculate scores for all validation triples from the dataset we trained the model upon.

>>> from pykeen.datasets import get_dataset
>>> from pykeen.predict import predict_triples
>>> dataset = get_dataset(dataset="nations")
>>> pack = predict_triples(model=result.model, triples=dataset.validation)

The variable :data:`pack` now contains a :class:`pykeen.predict.ScorePack`, which essentially is a pair of
ID-based triples with their predicted scores. For interpretation, it can be helpful to add their corresponding labels,
which the `"nations"` dataset offers, and convert them to a pandas dataframe:

>>> df = pack.process(factory=result.training).df

Since we now have a dataframe, we can utilize the full power of pandas for our subsequent analysis, e.g., showing the
triples which received the highest score

>>> df.nlargest(n=5, columns="score")

or investigate whether certain entities generally receive larger scores

>>> df.groupby(by=["head_id", "head_label"]).agg({"score": ["mean", "std", "count"]})


Target Scoring
--------------

:func:`pykeen.predict.predict_target`'s primary usecase is link prediction or relation prediction.
For instance, we could use our models to score all possible tail entities for the query `("uk", "conferences", ?)` via

>>> from pykeen.datasets import get_dataset
>>> from pykeen.predict import predict_target
>>> dataset = get_dataset(dataset="nations")
>>> pred = predict_target(
...     model=result.model,
...     head="uk",
...     relation="conferences",
...     triples_factory=result.training,
... )

Notice that the result stored into `pred` is a :class:`pykeen.predict.Predictions` object, which offers some
post-processing options. For instance, we can remove all targets which are already know from the training set

>>> pred_filtered = pred.filter_triples(dataset.training)

or add additional columns to the dataframe proving the information whether the target is contained in another set,
e.g., the validation or testing set.

>>> pred_annotated = pred_filtered.add_membership_columns(validation=dataset.validation, testing=dataset.testing)

The predictions object also exposes filtered / annotated dataframe through its `df` attribute

>>> pred_annotated.df

Full Scoring
------------
Finally, we can use :func:`pykeen.predict.predict` to calculate scores for *all* possible triples. Notice that
this operation can be prohibitively expensive for reasonably sized knowledge graphs, and the model may produce
additional ill-calibrated scores for entity/relation combinations it has never seen paired before during training.
The next line calculates *and* stores all triples and scores

>>> from pykeen.predict import predict_all
>>> pack = predict_all(model=result.model)

In addition to the expensive calculations, this additionally requires us to have sufficient memory available to store
all scores. A computationally equally expensive option with reduced, fixed memory requirement is to store only
the triples with the top $k$ scores. This can be done through the optional parameter `k`

>>> pack = predict_all(model=result.model, k=10)

We can again convert the score pack to a predictions object for further filtering, e.g., adding a column indicating
whether the triple has been seen during training

>>> pred = pack.process(factory=result.training)
>>> pred_annotated = pred.add_membership_columns(training=result.training)
>>> pred_annotated.df


Low-Level
=========
The following section outlines some details about the implementation of operations
which require calculating scores for all triples. The algorithm works are follows:

.. code-block:: python

  for batch in DataLoader(dataset, batch_size=batch_size):
    scores = model.predict(batch)
    for consumer in consumers:
      consumer(batch, scores)

Here, `dataset` is a :class:`pykeen.predict.PredictionDataset`, which breaks
the score calculation down into individual target predictions (e.g., tail predictions).
Implementations include :class:`pykeen.predict.AllPredictionDataset` and
:class:`pykeen.predict.PartiallyRestrictedPredictionDataset`. Notice that the
prediction tasks are built lazily, i.e., only instantiating the prediction tasks when
accessed. Moreover, the :mod:`torch_max_mem` package is used to automatically tune the
batch size to maximize the memory utilization of the hardware at hand.

For each batch, the scores of the prediction task are calculated once. Afterwards, multiple
*consumers* can process these scores. A consumer extends :class:`pykeen.predict.ScoreConsumer`
and receives the batch, i.e., input to the predict method, as well as the tensor of predicted scores.
Examples include

- :class:`pykeen.predict.CountScoreConsumer`: a simple consumer which only counts how many scores
  it has seen. Mostly used for debugging or testing purposes
- :class:`pykeen.predict.AllScoreConsumer`: accumulates all scores into a single huge tensor.
  This incurs massive memory requirements for reasonably sized datasets, and often can be avoided by
  interleaving the processing of the scores with calculation of individual batches.
- :class:`pykeen.predict.TopKScoreConsumer`: keeps only the top $k$ scores as well as the inputs
  leading to them. This is a memory-efficient variant of first accumulating all scores, then sorting by
  score and keeping only the top entries.

Potential Caveats
=================
The model is trained on a particular link prediction task, e.g. to predict the appropriate tail for a
given head/relation pair. This means that while the model can technically also predict other links, e.g.,
relations between a given head/tail pair, it must be done with the caveat that it was not
trained for this task, and thus its scores may behave unexpectedly.

Migration Guide
===============
Until version 1.9, the model itself provided wrappers which would delegate to the corresponding method
in `pykeen.models.predict`

* `model.get_all_prediction_df`
* `model.get_prediction_df`
* `model.get_head_prediction_df`
* `model.get_relation_prediction_df`
* `model.get_tail_prediction_df`

These methods were already deprecated and could be replaced by providing the model as explicit parameter
to the stand-alone functions from the prediction module. Thus, we will focus on the migrating the
stand-alone functions.

In the `pykeen.models.predict` module, the prediction methods were organized differently. There were

* `get_prediction_df`
* `get_head_prediction_df`
* `get_relation_prediction_df`
* `get_tail_prediction_df`
* `get_all_prediction_df`
* `predict_triples_df`

where `get_head_prediction_df`, `get_relation_prediction_df` and `get_tail_prediction_df` were deprecated in favour
of directly using `get_prediction_df` with all but the prediction target being provided, i.e., e.g.,

>>> from pykeen.models import predict
>>> prediction.get_tail_prediction_df(
...     model=model,
...     head_label="belgium",
...     relation_label="locatedin",
...     triples_factory=result.training,
... )

was deprecated in favour of

>>> from pykeen.models import predict
>>> predict.get_prediction_df(
...     model=model,
...     head_label="brazil",
...     relation_label="intergovorgs",
...     triples_factory=result.training,
... )


`get_prediction_df`
-------------------

The old use of

>>> from pykeen.models import predict
>>> predict.get_prediction_df(
...     model=model,
...     head_label="brazil",
...     relation_label="intergovorgs",
...     triples_factory=result.training,
... )

can be replaced by

>>> from pykeen import predict
>>> predict.predict_target(
...     model=model,
...     head="brazil",
...     relation="intergovorgs",
...     triples_factory=result.training,
... ).df

Notice the trailing `.df`.

`get_all_prediction_df`
-----------------------

The old use of

>>> from pykeen.models import predict
>>> predictions_df = predict.get_all_prediction_df(model, triples_factory=result.training)

can be replaced by

>>> from pykeen import predict
>>> predict.predict_all(model=model).process(factory=result.training).df

`predict_triples_df`
--------------------

The old use of

>>> from pykeen.models import predict
>>> score_df = predict.predict_triples_df(
...     model=model,
...     triples=[("brazil", "conferences", "uk"), ("brazil", "intergovorgs", "uk")],
...     triples_factory=result.training,
... )

can be replaced by

>>> from pykeen import predict
>>> score_df = predict.predict_triples(
...     model=model,
...     triples=[("brazil", "conferences", "uk"), ("brazil", "intergovorgs", "uk")],
...     triples_factory=result.training,
... )
"""

import collections
import dataclasses
import logging
import math
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Collection, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy
import pandas
import torch
import torch.utils.data
from torch_max_mem import maximize_memory_utilization
from tqdm.auto import tqdm
from typing_extensions import TypeAlias  # Python <=3.9

from .constants import COLUMN_LABELS, TARGET_TO_INDEX
from .models.base import Model
from .triples import AnyTriples, CoreTriplesFactory, TriplesFactory, get_mapped_triples
from .triples.utils import tensor_to_df
from .typing import (
    LABEL_HEAD,
    LABEL_RELATION,
    LABEL_TAIL,
    DeviceHint,
    InductiveMode,
    LabeledTriples,
    MappedTriples,
    Target,
)
from .utils import invert_mapping, isin_many_dim, resolve_device

__all__ = [
    # high-level
    "predict_all",
    "predict_triples",
    "predict_target",
    # Low-Level
    "consume_scores",
    "ScoreConsumer",
    "CountScoreConsumer",
    "TopKScoreConsumer",
    "AllScoreConsumer",
    "CountScoreConsumer",
    "ScorePack",
    "Predictions",
    "TriplePredictions",
    "TargetPredictions",
    "PredictionDataset",
    "AllPredictionDataset",
    "PartiallyRestrictedPredictionDataset",
]

logger = logging.getLogger(__name__)


# cf. https://github.com/python/mypy/issues/5374
@dataclasses.dataclass  # type: ignore
class Predictions(ABC):
    """Base class for predictions."""

    #: the dataframe; has to have a column named "score"
    df: pandas.DataFrame

    #: an optional factory to use for labeling
    factory: Optional[CoreTriplesFactory]

    def __post_init__(self):
        """Verify constraints."""
        if "score" not in self.df.columns:
            raise ValueError(f"df must have a column named 'score', but df.columns={self.df.columns}")

    def exchange_df(self, df: pandas.DataFrame) -> "Predictions":
        """Create a copy of the object with its dataframe exchanged."""
        return self.__class__(**collections.ChainMap(dict(df=df), dataclasses.asdict(self)))

    @abstractmethod
    def _contains(self, df: pandas.DataFrame, mapped_triples: MappedTriples, invert: bool = False) -> numpy.ndarray:
        """
        Return which of the rows of the given data frame are contained in the ID-based triples.

        :param df: nrows: n
            the predictions
        :param mapped_triples: shape: (m, 3)
            the ID-based triples
        :param invert:
            whether to invert the result

        :return: shape: (n,), dtype: bool
            a boolean mask indicating which row is contained in the given ID-based triples
        """
        raise NotImplementedError

    def filter_triples(self, *triples: Optional[AnyTriples]) -> pandas.DataFrame:
        """Filter out known triples."""
        df = self.df
        for mapped_triples in triples:
            if mapped_triples is None:
                continue
            df = df[
                self._contains(
                    df=df, mapped_triples=get_mapped_triples(mapped_triples, factory=self.factory), invert=True
                )
            ]
        return self.exchange_df(df=df)

    def add_membership_columns(self, **filter_triples: Optional[AnyTriples]) -> pandas.DataFrame:
        """Add columns indicating whether the triples are known."""
        df = self.df.copy()
        for key, mapped_triples in filter_triples.items():
            if mapped_triples is None:
                continue
            df[f"in_{key}"] = self._contains(
                df=df, mapped_triples=get_mapped_triples(mapped_triples, factory=self.factory)
            )
        return self.exchange_df(df=df)


@dataclasses.dataclass
class TriplePredictions(Predictions):
    """Triples with their predicted scores."""

    # docstr-coverage: inherited
    def __post_init__(self):  # noqa: D105
        super().__post_init__()
        columns = set(f"{column}_id" for column in COLUMN_LABELS)
        if not columns.issubset(self.df.columns):
            raise ValueError(f"df must have a columns named {columns}, but df.columns={self.df.columns}")

    # docstr-coverage: inherited
    def _contains(
        self, df: pandas.DataFrame, mapped_triples: MappedTriples, invert: bool = False
    ) -> numpy.ndarray:  # noqa: D102
        contained = (
            isin_many_dim(
                elements=torch.as_tensor(
                    df[[f"{target}_id" for target, _ in sorted(TARGET_TO_INDEX.items(), key=itemgetter(1))]].values,
                    device=mapped_triples.device,
                ),
                test_elements=mapped_triples,
            )
            .cpu()
            .numpy()
        )
        if invert:
            return ~contained
        return contained


@dataclasses.dataclass
class TargetPredictions(Predictions):
    """Targets with their predicted scores."""

    #: the prediction target
    target: Target

    #: the other column's fixed IDs
    other_columns_fixed_ids: Tuple[int, int]

    # docstr-coverage: inherited
    def __post_init__(self):  # noqa: D105
        super().__post_init__()
        if f"{self.target}_id" not in self.df.columns:
            raise ValueError(f"df must have a column named '{self.target}_id', but df.columns={self.df.columns}")

    # docstr-coverage: inherited
    def _contains(
        self, df: pandas.DataFrame, mapped_triples: MappedTriples, invert: bool = False
    ) -> numpy.ndarray:  # noqa: D102
        col = TARGET_TO_INDEX[self.target]
        other_cols = sorted(set(range(mapped_triples.shape[1])).difference({col}))
        device = mapped_triples.device
        other_col_ids = torch.as_tensor(data=self.other_columns_fixed_ids, dtype=torch.long, device=device)
        filter_mask = (mapped_triples[:, other_cols] == other_col_ids[None, :]).all(dim=-1)
        known_ids = mapped_triples[filter_mask, col].unique()
        query_ids = torch.as_tensor(df[f"{self.target}_id"].to_numpy(), device=device)
        return torch.isin(elements=query_ids, test_elements=known_ids, assume_unique=True, invert=invert).cpu().numpy()


@dataclasses.dataclass
class ScorePack:
    """A pair of result triples and scores."""

    #: the ID-based triples, shape: (n, 3)
    result: torch.LongTensor

    #: the scores
    scores: torch.FloatTensor

    def process(self, factory: Optional[CoreTriplesFactory] = None, **kwargs) -> "TriplePredictions":
        """Start post-processing scores."""
        if factory is None:
            df = tensor_to_df(self.result, score=self.scores, **kwargs)
        else:
            df = factory.tensor_to_df(self.result, score=self.scores, **kwargs)
        return TriplePredictions(df=df, factory=factory)


def _get_targets(
    ids: Union[None, torch.Tensor, Collection[Union[str, int]]],
    triples_factory: Optional[TriplesFactory],
    device: torch.device,
    entity: bool = True,
) -> Tuple[Optional[Iterable[str]], Optional[Iterable[int]], Optional[torch.Tensor]]:
    """
    Prepare prediction targets for restricted target prediction.

    :param ids:
        the target IDs in any of the supported formats
    :param triples_factory:
        the triples factory used to obtain labels. Must be provided, if any of the `ids` is given as a string
    :param device:
        the device to move the tensor to
    :param entity:
        whether the prediction target is an entity or relation

    :raises ValueError:
        if any of the ids is given as string, but no triples factory is available for conversion to IDs.

    :return:
        a 3-tuple of an optional list of labels, a list of ids, and the tensor to pass to the prediction method.
    """
    # 3-tuple for return
    labels: Optional[Iterable[str]] = None
    id_list: Optional[Iterable[int]] = None
    tensor: Optional[torch.Tensor] = None

    # extract label information, if possible
    label_to_id: Optional[Mapping[str, int]]
    id_to_label: Optional[Mapping[int, str]]
    if isinstance(triples_factory, TriplesFactory):
        label_to_id = triples_factory.entity_to_id if entity else triples_factory.relation_to_id
        id_to_label = invert_mapping(label_to_id)
    else:
        id_to_label = label_to_id = None

    # no restriction
    if ids is None:
        if id_to_label is not None:
            labels = map(itemgetter(1), sorted(id_to_label.items()))
    elif isinstance(ids, torch.Tensor):
        # restriction is a tensor
        tensor = ids.to(device)
        id_list = ids.tolist()
    else:
        # restriction is a sequence of integers or strings
        if not all(isinstance(i, int) for i in ids):
            if label_to_id is None:
                raise ValueError(
                    "If any of the ids is given as string, a triples factory must be provided for ID conversion."
                )
            ids = [i if isinstance(i, int) else label_to_id[i] for i in ids]
        # now, restriction is a sequence of integers
        assert all(isinstance(i, int) for i in ids)
        id_list = sorted(ids)  # type: ignore
        tensor = torch.as_tensor(id_list, dtype=torch.long, device=device)
    # if explicit ids have been given, and label information is available, extract list of labels
    if id_list is not None and id_to_label is not None:
        labels = map(id_to_label.__getitem__, id_list)
    return labels, id_list, tensor


def _get_input_batch(
    factory: Optional[TriplesFactory] = None,
    # exactly one of them is None
    head: Union[None, int, str] = None,
    relation: Union[None, int, str] = None,
    tail: Union[None, int, str] = None,
) -> Tuple[Target, torch.LongTensor, Tuple[int, int]]:
    """Prepare input batch for prediction.

    :param factory:
        the triples factory used to translate labels to ids.
    :param head:
        the head entity
    :param relation:
        the relation
    :param tail:
        the tail entity

    :raises ValueError:
        if not exactly one of {head, relation, tail} is None

    :return:
        a 3-tuple (target, batch, batch_tuple) of the prediction target, the input batch, and the input batch as tuple.
    """
    # create input batch
    batch_ids: List[int] = []
    target: Optional[Target] = None
    if head is None:
        target = LABEL_HEAD
    else:
        if not isinstance(head, int):
            if factory is None:
                raise ValueError("If head is not given as index, a triples factory must be passed.")
            head = factory.entity_to_id[head]
        batch_ids.append(head)
    if relation is None:
        target = LABEL_RELATION
    else:
        if not isinstance(relation, int):
            if factory is None:
                raise ValueError("If relation is not given as index, a triples factory must be passed.")
            relation = factory.relation_to_id[relation]
        batch_ids.append(relation)
    if tail is None:
        target = LABEL_TAIL
    else:
        if not isinstance(tail, int):
            if factory is None:
                raise ValueError("If tail is not given as index, a triples factory must be passed.")
            tail = factory.entity_to_id[tail]
        batch_ids.append(tail)
    if target is None or len(batch_ids) != 2:
        raise ValueError(
            f"Exactly one of {{head, relation, tail}} must be None, but got {head}, {relation}, {tail}",
        )

    batch = cast(torch.LongTensor, torch.as_tensor([batch_ids], dtype=torch.long))
    return target, batch, (batch_ids[0], batch_ids[1])


# note type alias annotation required,
# cf. https://mypy.readthedocs.io/en/stable/common_issues.html#variables-vs-type-aliases
# batch, TODO: ids?
PredictionBatch: TypeAlias = torch.LongTensor


class ScoreConsumer:
    """A consumer of scores for visitor pattern."""

    result: torch.LongTensor
    scores: torch.FloatTensor
    flatten: bool

    @abstractmethod
    def __call__(
        self,
        batch: PredictionBatch,
        target: Target,
        scores: torch.FloatTensor,
    ) -> None:
        """Consume scores for the given hr_batch."""
        raise NotImplementedError

    def finalize(self) -> ScorePack:
        """Finalize the result to build a score pack."""
        return _build_pack(result=self.result, scores=self.scores, flatten=self.flatten)


class CountScoreConsumer(ScoreConsumer):
    """A simple consumer which counts the number of batches and scores."""

    def __init__(self) -> None:
        """Initialize the consumer."""
        super().__init__()
        self.batch_count = 0
        self.score_count = 0

    # docstr-coverage: inherited
    def __call__(
        self,
        batch: PredictionBatch,
        target: Target,
        scores: torch.FloatTensor,
    ) -> None:  # noqa: D102
        self.batch_count += batch.shape[0]
        self.score_count += scores.numel()


class TopKScoreConsumer(ScoreConsumer):
    """Collect top-k triples & scores."""

    flatten = False

    def __init__(self, k: int = 3, device: DeviceHint = None) -> None:
        """
        Initialize the consumer.

        :param k:
            the number of top-scored triples to collect
        :param device:
            the model's device
        """
        self.k = k
        device = resolve_device(device=device)
        # initialize buffer on device
        self.result = torch.empty(0, 3, dtype=torch.long, device=device)
        self.scores = torch.empty(0, device=device)

    # docstr-coverage: inherited
    def __call__(
        self,
        batch: PredictionBatch,
        target: Target,
        scores: torch.FloatTensor,
    ) -> None:  # noqa: D102
        batch_size, num_scores = scores.shape
        assert batch.shape == (batch_size, 2)

        # reshape, shape: (batch_size * num_entities,)
        top_scores = scores.view(-1)

        # get top scores within batch
        if top_scores.numel() >= self.k:
            top_scores, top_indices = top_scores.topk(
                k=min(self.k, top_scores.numel()),
                largest=True,
                sorted=False,
            )
            # determine corresponding indices
            # batch_id, score_id = divmod(top_indices, num_scores)
            batch_id = torch.div(top_indices, num_scores, rounding_mode="trunc")
            score_id = top_indices % num_scores
            key_indices = batch[batch_id]
        else:
            key_indices = batch.unsqueeze(dim=1).repeat(1, num_scores, 1).view(-1, 2)
            score_id = torch.arange(num_scores, device=batch.device).view(1, -1).repeat(batch_size, 1).view(-1)

        # combine to top triples
        j = 0
        triples = []
        for col in COLUMN_LABELS:
            if col == target:
                index = score_id
            else:
                index = key_indices[:, j]
                j += 1
            triples.append(index)
        top_triples = torch.stack(triples, dim=-1)

        # append to global top scores
        self.scores = torch.cat([self.scores, top_scores])
        self.result = torch.cat([self.result, top_triples])

        # reduce size if necessary
        if self.result.shape[0] > self.k:
            self.scores, indices = self.scores.topk(k=self.k, largest=True, sorted=False)
            self.result = self.result[indices]


class AllScoreConsumer(ScoreConsumer):
    """Collect scores for all triples."""

    flatten = True

    def __init__(self, num_entities: int, num_relations: int) -> None:
        """
        Initialize the consumer.

        :param num_entities:
            the number of entities
        :param num_relations:
            the number of relations
        """
        assert num_entities**2 * num_relations < (2**63 - 1)
        # initialize buffer on cpu
        self.scores = torch.empty(num_entities, num_relations, num_entities, device="cpu")
        # Explicitly create triples
        self.result = torch.stack(
            [
                torch.arange(num_entities).view(-1, 1, 1).repeat(1, num_relations, num_entities),
                torch.arange(num_relations).view(1, -1, 1).repeat(num_entities, 1, num_entities),
                torch.arange(num_entities).view(1, 1, -1).repeat(num_entities, num_relations, 1),
            ],
            dim=-1,
        ).view(-1, 3)

    # docstr-coverage: inherited
    def __call__(
        self,
        batch: PredictionBatch,
        target: Target,
        scores: torch.FloatTensor,
    ) -> None:  # noqa: D102
        j = 0
        selectors: List[Union[slice, torch.LongTensor]] = []
        for col in COLUMN_LABELS:
            if col == target:
                selector = slice(None)
            else:
                selector = batch[:, j]
                j += 1
            selectors.append(selector)
        if target == LABEL_HEAD:
            scores = scores.t()
        self.scores[selectors[0], selectors[1], selectors[2]] = scores.to(self.scores.device)


class PredictionDataset(torch.utils.data.Dataset):
    """A base class for prediction datasets."""

    def __init__(self, target: Target = LABEL_TAIL) -> None:
        """Initialize the dataset.

        :param target:
            the prediction target to use. Prefer targets which are efficient to predict with the given model,
            e.g., tails for ConvE.
        """
        super().__init__()
        # TODO: variable targets across batches/samples?
        self.target = target

    # docstr-coverage: inherited
    @abstractmethod
    def __getitem__(self, item: int) -> PredictionBatch:  # noqa: D105
        raise NotImplementedError

    # docstr-coverage: inherited
    @abstractmethod
    def __len__(self) -> int:  # noqa: D105
        raise NotImplementedError


class AllPredictionDataset(PredictionDataset):
    """A dataset for predicting all possible triples."""

    def __init__(self, num_entities: int, num_relations: int, **kwargs) -> None:
        """Initialize the dataset.

        :param num_entities:
            the number of entities
        :param num_relations:
            the number of relations
        :param kwargs:
            additional keyword-based parameters passed to :meth:`PredictionDataset.__init__`
        """
        super().__init__(**kwargs)
        self.num_entities = num_entities
        self.num_relations = num_relations
        # (?, r, t) => r.stride > t.stride
        # (h, ?, t) => h.stride > t.stride
        # (h, r, ?) => h.stride > r.stride
        self.divisor = num_relations if self.target == LABEL_TAIL else num_entities

    # docstr-coverage: inherited
    def __len__(self) -> int:  # noqa: D105
        if self.target == LABEL_RELATION:
            return self.num_entities**2
        return self.num_entities * self.num_relations

    # docstr-coverage: inherited
    def __getitem__(self, item: int) -> torch.LongTensor:  # noqa: D105
        quotient, remainder = divmod(item, self.divisor)
        return torch.as_tensor([quotient, remainder])


Restriction = Union[torch.LongTensor, Collection[int], int]


class PartiallyRestrictedPredictionDataset(PredictionDataset):
    r"""
    A dataset for scoring some links.

    "Some links" is defined as

    .. math ::
        \mathcal{T}_{interest} = \mathcal{E}_{h} \times \mathcal{R}_{r} \times \mathcal{E}_{t}

    .. note ::
        For now, the target, i.e., position whose prediction method in the model is utilized,
        must be the full set of entities/relations.

    Example
    .. code-block:: python

        # train model; note: needs larger number of epochs to do something useful ;-)
        from pykeen.pipeline import pipeline
        result = pipeline(dataset="nations", model="mure", training_kwargs=dict(num_epochs=0))

        # create prediction dataset, where the head entities is from a set of European countries,
        # and the relations are connected to tourism
        from pykeen.predict import PartiallyRestrictedPredictionDataset
        heads = result.training.entities_to_ids(entities=["netherlands", "poland", "uk"])
        relations = result.training.relations_to_ids(relations=["reltourism", "tourism", "tourism3"])
        dataset = PartiallyRestrictedPredictionDataset(heads=heads, relations=relations)

        # calculate all scores for this restricted set, and keep k=3 largest
        from pykeen.predict import consume_scores, TopKScoreConsumer
        consumer = TopKScoreConsumer(k=3)
        consume_scores(result.model, ds, consumer)
        score_pack = consumer.finalize()

        # add labels
        df = result.training.tensor_to_df(score_pack.result, score=score_pack.scores)
    """

    #: the choices for the first and second component of the input batch
    parts: Tuple[torch.LongTensor, torch.LongTensor]

    def __init__(
        self,
        *,
        heads: Optional[Restriction] = None,
        relations: Optional[Restriction] = None,
        tails: Optional[Restriction] = None,
        target: Target = LABEL_TAIL,
    ) -> None:
        """
        Initialize restricted prediction dataset.

        :param heads:
            the restricted head entities
        :param relations:
            the restricted relations
        :param tails:
            the restricted tails
        :param target:
            the prediction target

        :raises NotImplementedError:
            if the target position is restricted, or any non-target position is not restricted
        """
        super().__init__(target=target)
        parts: List[torch.LongTensor] = []
        for restriction, on in zip((heads, relations, tails), COLUMN_LABELS):
            if on == target:
                if restriction is not None:
                    raise NotImplementedError("Restrictions on the target are not yet supported.")
                continue
            if restriction is None:
                raise NotImplementedError("Requires size info")
            elif isinstance(restriction, int):
                restriction = [restriction]
            restriction = torch.as_tensor(restriction)
            parts.append(restriction)
        assert len(parts) == 2
        self.parts = (parts[0], parts[1])  # for mypy

    # docstr-coverage: inherited
    def __len__(self) -> int:  # noqa: D105
        return math.prod(map(len, self.parts))

    # docstr-coverage: inherited
    def __getitem__(self, item: int) -> PredictionBatch:  # noqa: D105
        remainder, quotient = divmod(item, len(self.parts[0]))
        return torch.as_tensor([self.parts[0][quotient], self.parts[1][remainder]])


@torch.inference_mode()
@maximize_memory_utilization(parameter_name="batch_size", keys=["model", "dataset", "consumers", "mode"])
def consume_scores(
    model: Model,
    dataset: PredictionDataset,
    *consumers: ScoreConsumer,
    batch_size: int = 1,
    mode: Optional[InductiveMode] = None,
) -> None:
    """
    Batch-wise calculation of all triple scores and consumption.

    From a high-level perspective, this method does the following:

    .. code-block:: python

        for batch in dataset:
            scores = model.predict(batch)
            for consumer in consumers:
                consumer(batch, scores)

    By bringing custom prediction datasets and/or score consumers, this method is highly configurable.

    :param model:
        the model used to calculate scores
    :param dataset:
        the dataset defining the prediction tasks, i.e., inputs to `model.predict` to loop over.
    :param consumers:
        the consumers of score batches
    :param batch_size:
        the batch size to use. Will automatically be lowered, if the hardware cannot handle this large batch sizes
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.

    :raises ValueError:
        if no score consumers are given
    """
    if not consumers:
        raise ValueError("Did not receive any consumer")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    for batch in tqdm(data_loader, desc="scoring", unit="batch", unit_scale=True, leave=False):
        batch = batch.to(model.device)
        # calculate batch scores onces
        scores = model.predict(batch, target=dataset.target, full_batch=False, mode=mode)
        # consume by all consumers
        for consumer in consumers:
            consumer(batch, target=dataset.target, scores=scores)


def _build_pack(result: torch.LongTensor, scores: torch.FloatTensor, flatten: bool = False) -> ScorePack:
    """Sort final result and package in a score pack."""
    scores, indices = torch.sort(scores.flatten() if flatten else scores, descending=True)
    result = result[indices]
    return ScorePack(result=result, scores=scores)


@maximize_memory_utilization(keys=["model"])
def _predict_triples_batched(
    model: Model,
    mapped_triples: MappedTriples,
    batch_size: int,
    *,
    mode: Optional[InductiveMode],
) -> torch.FloatTensor:
    """Predict scores for triples in batches."""
    return torch.cat(
        [
            model.predict_hrt(hrt_batch=hrt_batch, mode=mode)
            for hrt_batch in mapped_triples.split(split_size=batch_size, dim=0)
        ],
        dim=0,
    )


# TODO: Support partial dataset
@torch.inference_mode()
def predict_all(
    model: Model,
    *,
    k: Optional[int] = None,
    batch_size: Optional[int] = 1,
    mode: Optional[InductiveMode] = None,
    target: Target = LABEL_TAIL,
) -> ScorePack:
    """Calculate scores for all triples, and either keep all of them or only the top k triples.

    :param model:
        A PyKEEN model
    :param k:
        The number of triples to return. Set to ``None`` to keep all.
    :param batch_size:
        The batch size to use for calculating scores; set to `None` to determine largest possible batch size
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.
    :param target:
        the prediction target to use. Prefer targets which are efficient to predict with the given model,
        e.g., tails for ConvE.

    :return:
        A score pack of parallel triples and scores
    """
    # note: the models' predict method takes care of setting the model to evaluation mode
    logger.warning(
        f"predict is an expensive operation, involving {model.num_entities ** 2 * model.num_real_relations:,} "
        f"score evaluations.",
    )

    consumer: ScoreConsumer
    if k is None:
        logger.warning(
            "Not providing k to `predict_all` entails huge memory requirements for reasonably-sized knowledge graphs.",
        )
        consumer = AllScoreConsumer(num_entities=model.num_entities, num_relations=model.num_relations)
    else:
        consumer = TopKScoreConsumer(k=k, device=model.device)
    dataset = AllPredictionDataset(
        num_entities=model.num_entities, num_relations=model.num_real_relations, target=target
    )
    consume_scores(model, dataset, consumer, batch_size=batch_size or len(dataset), mode=mode)
    return consumer.finalize()


@torch.inference_mode()
def predict_target(
    model: Model,
    *,
    # exactly one of them is None
    head: Union[None, int, str] = None,
    relation: Union[None, int, str] = None,
    tail: Union[None, int, str] = None,
    #
    triples_factory: Optional[TriplesFactory] = None,
    targets: Union[None, torch.LongTensor, Sequence[Union[int, str]]] = None,
    mode: Optional[InductiveMode] = None,
) -> Predictions:
    """Get predictions for the head, relation, and/or tail combination.

    .. note ::
        Exactly one of `head`, `relation` and `tail` should be None. This is the position
        which will be predicted.

    :param model:
        A PyKEEN model

    :param head:
        the head entity, either as ID or as label. If None, predict heads
    :param relation:
        the relation, either as ID or as label. If None, predict relations
    :param tail:
        the tail entity, either as ID or as label. If None, predict tails

    :param targets:
        restrict prediction to these targets. `None` means no restriction, i.e., scoring all entities/relations.
    :param triples_factory:
        the training triples factory; required if head/relation/tail are given as string, and used to translate the
        label to an ID.

    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.

    :return:
        The predictions, containing either the $k$ highest scoring targets, or all targets if $k$ is `None`.
    """
    # note: the models' predict method takes care of setting the model to evaluation mode

    # get input & target
    target, batch, other_col_ids = _get_input_batch(factory=triples_factory, head=head, relation=relation, tail=tail)

    # get label-to-id mapping and prediction targets
    labels, ids, targets = _get_targets(
        ids=targets, triples_factory=triples_factory, device=model.device, entity=relation is not None
    )

    # get scores
    scores = model.predict(batch, full_batch=False, mode=mode, ids=targets, target=target).squeeze(dim=0).tolist()
    if ids is None:
        ids = range(len(scores))

    # create raw dataframe
    data = {f"{target}_id": ids, "score": scores}
    if labels is not None:
        data[f"{target}_label"] = labels
    df = pandas.DataFrame(data=data).sort_values("score", ascending=False)
    return TargetPredictions(df=df, factory=triples_factory, target=target, other_columns_fixed_ids=other_col_ids)


@torch.inference_mode()
def predict_triples(
    model: Model,
    *,
    triples: Union[None, MappedTriples, LabeledTriples, Union[Tuple[str, str, str], Sequence[Tuple[str, str, str]]]],
    triples_factory: Optional[CoreTriplesFactory] = None,
    batch_size: Optional[int] = None,
    mode: Optional[InductiveMode] = None,
) -> ScorePack:
    """
    Predict on labeled or mapped triples.

    :param model:
        The model.
    :param triples: shape: (num_triples, 3)
        The triples in one of the following formats:

        - A single label-based triple.
        - A list of label-based triples.
        - An array of label-based triples
        - An array of ID-based triples.
        - None. In this case, a triples factory has to be provided, and its triples will be used.

    :param triples_factory:
        The triples factory. Must be given if triples are label-based. If provided and triples are ID-based, add labels
        to result.
    :param batch_size:
        The batch size to use. Use None for automatic memory optimization.
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.

    :return:
        a score pack of the triples with the predicted scores.
    """
    # note: the models' predict method takes care of setting the model to evaluation mode
    # normalize input
    triples = get_mapped_triples(triples, factory=triples_factory)
    # calculate scores (with automatic memory optimization)
    scores = _predict_triples_batched(
        model=model, mapped_triples=triples, batch_size=batch_size or len(triples), mode=mode
    ).squeeze(dim=1)
    return ScorePack(result=triples, scores=scores)
