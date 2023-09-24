# -*- coding: utf-8 -*-

"""Utility classes for constructing inductive datasets."""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Union

from pystow.utils import download, name_from_url
from tabulate import tabulate

from ...constants import PYKEEN_DATASETS
from ...triples import CoreTriplesFactory, TriplesFactory
from ...utils import normalize_path

__all__ = [
    # Base class
    "InductiveDataset",
    # Mid-level classes
    "EagerInductiveDataset",
    "LazyInductiveDataset",
    "DisjointInductivePathDataset",
    "UnpackedRemoteDisjointInductiveDataset",
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
    inductive_validation: Optional[CoreTriplesFactory] = None
    #: All datasets should take care of inverse triple creation
    create_inverse_triples: bool = True

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
            )
        ]

    def summary_str(self, title: Optional[str] = None, show_examples: Optional[int] = 5, end="\n") -> str:
        """Make a summary string of all of the factories."""
        rows = self._summary_rows()
        n_triples = sum(count for *_, count in rows)
        rows.append(("Total", "-", "-", n_triples))
        t = tabulate(rows, headers=["Name", "Entities", "Relations", "Triples"])
        rv = f"{title or self.__class__.__name__} (create_inverse_triples={self.create_inverse_triples})\n{t}"
        if show_examples:
            if not isinstance(self.transductive_training, TriplesFactory):
                raise AttributeError(f"{self.transductive_training.__class__} does not have labeling information.")
            examples = tabulate(
                self.transductive_training.label_triples(self.transductive_training.mapped_triples[:show_examples]),
                headers=["Head", "Relation", "tail"],
            )
            rv += "\n" + examples
        return rv + end

    def summarize(self, title: Optional[str] = None, show_examples: Optional[int] = 5, file=None) -> None:
        """Print a summary of the dataset."""
        print(self.summary_str(title=title, show_examples=show_examples), file=file)  # noqa:T201

    def __str__(self) -> str:  # noqa: D105
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
    inductive_validation: Optional[CoreTriplesFactory] = None
    create_inverse_triples: bool = True


class LazyInductiveDataset(InductiveDataset):
    """An inductive dataset that has lazy loading."""

    #: The actual instance of the training factory, which is exposed to the user through `transductive_training`
    _transductive_training: Optional[TriplesFactory] = None
    #: The actual instance of the inductive inference factory,
    #: which is exposed to the user through `inductive_inference`
    _inductive_inference: Optional[TriplesFactory] = None
    #: The actual instance of the testing factory, which is exposed to the user through `inductive_testing`
    _inductive_testing: Optional[TriplesFactory] = None
    #: The actual instance of the validation factory, which is exposed to the user through `inductive_validation`
    _inductive_validation: Optional[TriplesFactory] = None
    #: The directory in which the cached data is stored
    cache_root: pathlib.Path

    @property
    def transductive_training(self) -> TriplesFactory:  # type:ignore # noqa: D401
        """The training triples factory."""
        if not self._loaded:
            self._load()
        assert self._transductive_training is not None
        return self._transductive_training

    @property
    def inductive_inference(self) -> TriplesFactory:  # type:ignore # noqa: D401
        """The inductive inference triples factory. MIGHT or MIGHT NOT share indices with the transductive train."""
        if not self._loaded:
            self._load()
        assert self._inductive_inference is not None
        return self._inductive_inference

    @property
    def inductive_testing(self) -> TriplesFactory:  # type:ignore # noqa: D401
        """The testing triples factory that share indices with the INDUCTIVE INFERENCE triples factory."""
        if not self._loaded:
            self._load()
        assert self._inductive_testing is not None
        return self._inductive_testing

    @property
    def inductive_validation(self) -> Optional[TriplesFactory]:  # type:ignore # noqa: D401
        """The validation triples factory that shares indices with the INDUCTIVE INFERENCE triples factory."""
        if not self._loaded:
            self._load()
        assert self._inductive_validation is not None
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
        cache_root: Union[None, str, pathlib.Path],
        version: Optional[str] = None,
        sep_train_inference: bool = False,
    ) -> pathlib.Path:
        """Get the appropriate cache root directory.

        :param cache_root: If none is passed, defaults to a subfolder of the
            PyKEEN home directory defined in :data:`pykeen.constants.PYKEEN_HOME`.
            The subfolder is named based on the class inheriting from
            :class:`pykeen.datasets.base.Dataset`.
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

    def _cache_sub_directories(self, version: Optional[str]) -> Iterable[str]:
        """Iterate over appropriate cache sub-directory."""
        # TODO: use class-resolver normalize?
        yield self.__class__.__name__.lower()
        # add v1 / v2 / v3 / v4 for inductive splits if available
        if version:
            yield version


class DisjointInductivePathDataset(LazyInductiveDataset):
    """A disjoint inductive dataset specified by paths.

    Contains a lazy reference to a training, inductive inference, inductive testing, and inductive validation dataset.
    In this dataset, inductive inference is disjoint with the transductive train
    """

    def __init__(
        self,
        transductive_training_path: Union[str, pathlib.Path],
        inductive_inference_path: Union[str, pathlib.Path],
        inductive_testing_path: Union[str, pathlib.Path],
        inductive_validation_path: Union[str, str, pathlib.Path],
        eager: bool = False,
        create_inverse_triples: bool = False,
        load_triples_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Initialize the dataset.

        :param transductive_training_path: Path to the training triples file or training triples file.
        :param inductive_inference_path: Path to the inductive inference triples file or training triples file.
        :param inductive_testing_path: Path to the testing triples file or testing triples file.
        :param inductive_validation_path: Path to the validation triples file or validation triples file.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param load_triples_kwargs: Arguments to pass through to :func:`TriplesFactory.from_path`
            and ultimately through to :func:`pykeen.triples.utils.load_triples`.
        """
        self.transductive_training_path = pathlib.Path(transductive_training_path)
        self.inductive_inference_path = pathlib.Path(inductive_inference_path)
        self.inductive_testing_path = pathlib.Path(inductive_testing_path)
        self.inductive_validation_path = pathlib.Path(inductive_validation_path)

        self.create_inverse_triples = create_inverse_triples
        self.load_triples_kwargs = load_triples_kwargs

        if eager:
            self._load()

    def _load(self) -> None:
        self._transductive_training = TriplesFactory.from_path(
            path=self.transductive_training_path,
            create_inverse_triples=self.create_inverse_triples,
            load_triples_kwargs=self.load_triples_kwargs,
        )

        # important: inductive_inference shares the same RELATIONS with the transductive training graph
        self._inductive_inference = TriplesFactory.from_path(
            path=self.inductive_inference_path,
            create_inverse_triples=self.create_inverse_triples,
            relation_to_id=self._transductive_training.relation_to_id,
            load_triples_kwargs=self.load_triples_kwargs,
        )

        # inductive validation shares both ENTITIES and RELATIONS with the inductive inference graph
        self._inductive_validation = TriplesFactory.from_path(
            path=self.inductive_validation_path,
            entity_to_id=self._inductive_inference.entity_to_id,  # shares entity index with inductive inference
            relation_to_id=self._inductive_inference.relation_to_id,  # shares relation index with inductive inference
            # do not explicitly create inverse triples for testing; this is handled by the evaluation code
            create_inverse_triples=False,
            load_triples_kwargs=self.load_triples_kwargs,
        )

        # inductive testing shares both ENTITIES and RELATIONS with the inductive inference graph
        self._inductive_testing = TriplesFactory.from_path(
            path=self.inductive_testing_path,
            entity_to_id=self._inductive_inference.entity_to_id,  # share entity index with inductive inference
            relation_to_id=self._inductive_inference.relation_to_id,  # share relation index with inductive inference
            # do not explicitly create inverse triples for testing; this is handled by the evaluation code
            create_inverse_triples=False,
            load_triples_kwargs=self.load_triples_kwargs,
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f'{self.__class__.__name__}(training_path="{self.transductive_training_path}", '
            f' inductive_inference="{self.inductive_inference_path}",'
            f' inductive_test="{self.inductive_testing_path}",'
            f' inductive_validation="{self.inductive_validation_path}")'
        )


class UnpackedRemoteDisjointInductiveDataset(DisjointInductivePathDataset):
    """A dataset with all four of train, inductive_inference, inductive test, and inductive validation sets as URLs."""

    def __init__(
        self,
        transductive_training_url: str,
        inductive_inference_url: str,
        inductive_testing_url: str,
        inductive_validation_url: str,
        cache_root: Optional[str] = None,
        force: bool = False,
        eager: bool = False,
        create_inverse_triples: bool = False,
        load_triples_kwargs: Optional[Mapping[str, Any]] = None,
        download_kwargs: Optional[Mapping[str, Any]] = None,
        version: Optional[str] = None,
    ):
        """Initialize dataset.

        :param transductive_training_url: The URL of the training file
        :param inductive_inference_url: The URL of the inductive inference graph file
        :param inductive_testing_url: The URL of the inductive testing file
        :param inductive_validation_url: The URL of the inductive validation file
        :param cache_root:
            An optional directory to store the extracted files. Is none is given, the default PyKEEN directory is used.
            This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to ``~/.data/pykeen``.
        :param force: If true, redownload any cached files
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param load_triples_kwargs: Arguments to pass through to :func:`TriplesFactory.from_path`
            and ultimately through to :func:`pykeen.triples.utils.load_triples`.
        :param download_kwargs: Keyword arguments to pass to :func:`pystow.utils.download`
        :param version: accepts a string "v1" to "v4" to select among Teru et al inductive datasets
        """
        self.cache_root = self._help_cache(cache_root, version, sep_train_inference=True)

        self.transductive_training_url = transductive_training_url
        self.inductive_inference_url = inductive_inference_url
        self.inductive_testing_url = inductive_testing_url
        self.inductive_validation_url = inductive_validation_url

        transductive_training_path = self.cache_root.joinpath("training", name_from_url(self.transductive_training_url))
        inductive_inference_path = self.cache_root.joinpath("inference", name_from_url(self.inductive_inference_url))
        inductive_testing_path = self.cache_root.joinpath("inference", name_from_url(self.inductive_testing_url))
        inductive_validation_path = self.cache_root.joinpath("inference", name_from_url(self.inductive_validation_url))

        download_kwargs = {} if download_kwargs is None else dict(download_kwargs)
        download_kwargs.setdefault("backend", "urllib")

        for url, path in [
            (self.transductive_training_url, transductive_training_path),
            (self.inductive_inference_url, inductive_inference_path),
            (self.inductive_testing_url, inductive_testing_path),
            (self.inductive_validation_url, inductive_validation_path),
        ]:
            if force or not path.is_file():
                download(url, path, **download_kwargs)

        super().__init__(
            transductive_training_path=transductive_training_path,
            inductive_inference_path=inductive_inference_path,
            inductive_testing_path=inductive_testing_path,
            inductive_validation_path=inductive_validation_path,
            eager=eager,
            create_inverse_triples=create_inverse_triples,
            load_triples_kwargs=load_triples_kwargs,
        )
