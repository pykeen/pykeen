"""Utility classes for constructing inductive datasets."""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Unpack

from pystow.utils import DownloadKwargs, name_from_url
from tabulate import tabulate

from ..base import PathDatasetKwargs
from ..source import RemoteSimpleSource, SimpleSource, Source
from ...constants import PYKEEN_DATASETS
from ...triples import CoreTriplesFactory, TriplesFactory
from ...utils import normalize_path

__all__ = [
    "DisjointInductivePathDataset",
    "DisjointInductiveSourceDataset",
    "EagerInductiveDataset",
    "InductiveDataset",
    "LazyInductiveDataset",
    "UnpackedRemoteDisjointInductiveDataset",
    "Version",
]

logger = logging.getLogger(__name__)


class InductiveDataset:
    """Contains transductive train and inductive inference/validation/test datasets."""

    #: A factory wrapping the training triples
    transductive_training: CoreTriplesFactory
    #: A factory wrapping the inductive inference triples that MIGHT or MIGHT NOT
    # share indices with the transductive training
    inductive_inference: CoreTriplesFactory
    #: A factory wrapping the testing triples, that share indices with the INDUCTIVE INFERENCE triples
    inductive_testing: CoreTriplesFactory
    #: A factory wrapping the validation triples, that share indices with the INDUCTIVE INFERENCE triples
    inductive_validation: CoreTriplesFactory | None = None

    def _summary_rows(self):
        return [
            (label, triples_factory.num_entities, triples_factory.num_relations, triples_factory.num_triples)
            for label, triples_factory in zip(
                ("Transductive Training", "Inductive Inference", "Inductive Testing", "Inductive Validation"),
                (
                    self.transductive_training,
                    self.inductive_inference,
                    self.inductive_testing,
                    self.inductive_validation,
                ),
                strict=True,
            )
            # note: the validation factory is optional
            if triples_factory is not None
        ]

    def summary_str(self, title: str | None = None, show_examples: int | None = 5, end="\n") -> str:
        """Make a summary string of all of the factories."""
        rows = self._summary_rows()
        n_triples = sum(count for *_, count in rows)
        rows.append(("Total", "-", "-", n_triples))
        t = tabulate(rows, headers=["Name", "Entities", "Relations", "Triples"])
        rv = f"{title or self.__class__.__name__}\n{t}"
        if show_examples:
            if not isinstance(self.transductive_training, TriplesFactory):
                raise AttributeError(f"{self.transductive_training.__class__} does not have labeling information.")
            examples = tabulate(
                self.transductive_training.label_triples(self.transductive_training.mapped_triples[:show_examples]),
                headers=["Head", "Relation", "tail"],
            )
            rv += "\n" + examples
        return rv + end

    def summarize(self, title: str | None = None, show_examples: int | None = 5, file=None) -> None:
        """Print a summary of the dataset."""
        print(self.summary_str(title=title, show_examples=show_examples), file=file)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(Training num_entities={self.transductive_training.num_entities},"
            f" num_relations={self.transductive_training.num_relations})"
        )


@dataclass
class EagerInductiveDataset(InductiveDataset):
    """An eager inductive datasets."""

    transductive_training: CoreTriplesFactory
    inductive_inference: CoreTriplesFactory
    inductive_testing: CoreTriplesFactory
    inductive_validation: CoreTriplesFactory | None = None


class LazyInductiveDataset(InductiveDataset):
    """An inductive dataset that has lazy loading."""

    #: The actual instance of the training factory, which is exposed to the user through `transductive_training`
    _transductive_training: TriplesFactory | None = None
    #: The actual instance of the inductive inference factory,
    #: which is exposed to the user through `inductive_inference`
    _inductive_inference: TriplesFactory | None = None
    #: The actual instance of the testing factory, which is exposed to the user through `inductive_testing`
    _inductive_testing: TriplesFactory | None = None
    #: The actual instance of the validation factory, which is exposed to the user through `inductive_validation`
    _inductive_validation: TriplesFactory | None = None
    #: The directory in which the cached data is stored
    cache_root: pathlib.Path

    @property
    def transductive_training(self) -> TriplesFactory:  # type: ignore[override]
        """The training triples factory."""
        if not self._loaded:
            self._load()
        if self._transductive_training is None:
            raise ValueError("self._transductive_training must not be None.")
        return self._transductive_training

    @property
    def inductive_inference(self) -> TriplesFactory:  # type: ignore[override]
        """The inductive inference triples factory. MIGHT or MIGHT NOT share indices with the transductive train."""
        if not self._loaded:
            self._load()
        if self._inductive_inference is None:
            raise ValueError("self._inductive_inference must not be None.")
        return self._inductive_inference

    @property
    def inductive_testing(self) -> TriplesFactory:  # type: ignore[override]
        """The testing triples factory that share indices with the INDUCTIVE INFERENCE triples factory."""
        if not self._loaded:
            self._load()
        if self._inductive_testing is None:
            raise ValueError("self._inductive_testing must not be None.")
        return self._inductive_testing

    @property
    def inductive_validation(self) -> TriplesFactory | None:  # type: ignore[override]
        """The validation triples factory that shares indices with the INDUCTIVE INFERENCE triples factory."""
        if not self._loaded:
            self._load()
        return self._inductive_validation

    @property
    def _loaded(self) -> bool:
        return self._transductive_training is not None and self._inductive_inference is not None

    def _load(self) -> None:
        raise NotImplementedError

    def _load_validation(self) -> None:
        raise NotImplementedError

    def _help_cache(
        self,
        cache_root: str | pathlib.Path | None,
        version: str | None = None,
        sep_train_inference: bool = False,
    ) -> pathlib.Path:
        """Get the appropriate cache root directory.

        :param cache_root: If none is passed, defaults to a subfolder of the PyKEEN home directory defined in
            :data:`~pykeen.constants.PYKEEN_HOME`. The subfolder is named based on the class inheriting from
            :class:`~pykeen.datasets.base.Dataset`.
        :param version: accepts a string "v1" to "v4" to select among Teru et al inductive datasets
        :param sep_train_inference: a flag to store training and inference splits in different folders

        :returns: A path object for the calculated cache root directory
        """
        cache_root = normalize_path(
            cache_root, *self._cache_sub_directories(version=version), default=PYKEEN_DATASETS, mkdir=True
        )
        if sep_train_inference:
            # generate subfolders 'training' and  'inference'
            for name in ("training", "inference"):
                cache_root.joinpath(name).mkdir(parents=True, exist_ok=True)
        logger.debug("using cache root at %s", cache_root.as_uri())
        return cache_root

    def _cache_sub_directories(self, version: str | None) -> Iterable[str]:
        """Iterate over appropriate cache sub-directory."""
        # TODO: use class-resolver normalize?
        yield self.__class__.__name__.lower()
        # add v1 / v2 / v3 / v4 for inductive splits if available
        if version:
            yield version


class DisjointInductiveSourceDataset(LazyInductiveDataset):
    """A disjoint inductive dataset specified by paths.

    Contains a lazy reference to a training, inductive inference, inductive testing, and inductive validation dataset.
    In this dataset, inductive inference is disjoint with the transductive train
    """

    def __init__(
        self,
        transductive_training_source: Source,
        inductive_inference_source: Source,
        inductive_testing_source: Source,
        inductive_validation_source: Source,
        **kwargs: Unpack[PathDatasetKwargs],
    ) -> None:
        """Initialize the dataset.

        :param transductive_training_source: The training triples source
        :param inductive_inference_source: The inductive inference triples source
        :param inductive_testing_source: The testing triples file source
        :param inductive_validation_source: The validation triples source
        """
        self.transductive_training_source = transductive_training_source
        self.inductive_inference_source = inductive_inference_source
        self.inductive_testing_source = inductive_testing_source
        self.inductive_validation_source = inductive_validation_source

        self.load_triples_kwargs = kwargs.get("load_triples_kwargs")

        if kwargs.get("eager"):
            self._load()

    def _load(self) -> None:
        with self.transductive_training_source.open() as file:
            self._transductive_training = TriplesFactory.from_path(
                file,
                load_triples_kwargs=self.load_triples_kwargs,
            )

        # important: inductive_inference shares the same RELATIONS with the transductive training graph
        with self.inductive_inference_source.open() as file:
            self._inductive_inference = TriplesFactory.from_path(
                file,
                relation_to_id=self._transductive_training.relation_to_id,
                load_triples_kwargs=self.load_triples_kwargs,
            )

        # inductive validation shares both ENTITIES and RELATIONS with the inductive inference graph
        with self.inductive_validation_source.open() as file:
            self._inductive_validation = TriplesFactory.from_path(
                file,
                # shares entity index with inductive inference
                entity_to_id=self._inductive_inference.entity_to_id,
                # shares relation index with inductive inference
                relation_to_id=self._inductive_inference.relation_to_id,
                load_triples_kwargs=self.load_triples_kwargs,
            )

        # inductive testing shares both ENTITIES and RELATIONS with the inductive inference graph
        with self.inductive_testing_source.open() as file:
            self._inductive_testing = TriplesFactory.from_path(
                file,
                # share entity index with inductive inference
                entity_to_id=self._inductive_inference.entity_to_id,
                # share relation index with inductive inference
                relation_to_id=self._inductive_inference.relation_to_id,
                load_triples_kwargs=self.load_triples_kwargs,
            )

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(training_path="{self.transductive_training_source}", '
            f' inductive_inference="{self.inductive_inference_source}",'
            f' inductive_test="{self.inductive_testing_source}",'
            f' inductive_validation="{self.inductive_validation_source}")'
        )


class DisjointInductivePathDataset(DisjointInductiveSourceDataset):
    """A disjoint inductive dataset specified by paths.

    Contains a lazy reference to a training, inductive inference, inductive testing, and inductive validation dataset.
    In this dataset, inductive inference is disjoint with the transductive train
    """

    def __init__(
        self,
        transductive_training_path: str | pathlib.Path,
        inductive_inference_path: str | pathlib.Path,
        inductive_testing_path: str | pathlib.Path,
        inductive_validation_path: str | pathlib.Path,
        **kwargs: Unpack[PathDatasetKwargs],
    ) -> None:
        """Initialize the dataset.

        :param transductive_training_path: Path to the training triples file or training triples file.
        :param inductive_inference_path: Path to the inductive inference triples file or training triples file.
        :param inductive_testing_path: Path to the testing triples file or testing triples file.
        :param inductive_validation_path: Path to the validation triples file or validation triples file.
        """
        super().__init__(
            transductive_training_source=SimpleSource(pathlib.Path(transductive_training_path)),
            inductive_inference_source=SimpleSource(pathlib.Path(inductive_inference_path)),
            inductive_testing_source=SimpleSource(pathlib.Path(inductive_testing_path)),
            inductive_validation_source=SimpleSource(pathlib.Path(inductive_validation_path)),
            **kwargs,
        )


#: The version for the ILP Teru dataset
Version = Literal["v1", "v2", "v3", "v4"]


class UnpackedRemoteDisjointInductiveDataset(DisjointInductiveSourceDataset):
    """A dataset with all four of train, inductive_inference, inductive test, and inductive validation sets as URLs."""

    def __init__(
        self,
        transductive_training_url: str,
        inductive_inference_url: str,
        inductive_testing_url: str,
        inductive_validation_url: str,
        *,
        cache_root: str | pathlib.Path | None = None,
        force: bool = False,
        download_kwargs: DownloadKwargs | None = None,
        version: Version | None = None,
        **kwargs: Unpack[PathDatasetKwargs],
    ) -> None:
        """Initialize dataset.

        :param transductive_training_url: The URL of the training file
        :param inductive_inference_url: The URL of the inductive inference graph file
        :param inductive_testing_url: The URL of the inductive testing file
        :param inductive_validation_url: The URL of the inductive validation file
        :param cache_root: An optional directory to store the extracted files. Is none is given, the default PyKEEN
            directory is used. This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to
            ``~/.data/pykeen``.
        :param force: If true, redownload any cached files
        :param download_kwargs: Keyword arguments to pass to :func:`pystow.utils.download`
        :param version: accepts a string "v1" to "v4" to select among Teru et al inductive datasets
        """
        cache_root_ = self._help_cache(cache_root, version, sep_train_inference=True)
        if download_kwargs is None:
            download_kwargs = {}
        download_kwargs.setdefault("backend", "urllib")

        def _get_source(directory: str, url: str) -> Source:
            return RemoteSimpleSource(
                path=cache_root_.joinpath(directory, name_from_url(url)),
                url=url,
                force=force,
                download_kwargs=download_kwargs,
            )

        super().__init__(
            transductive_training_source=_get_source("training", transductive_training_url),
            inductive_inference_source=_get_source("inference", inductive_inference_url),
            inductive_testing_source=_get_source("inference", inductive_testing_url),
            inductive_validation_source=_get_source("inference", inductive_validation_url),
            **kwargs,
        )
