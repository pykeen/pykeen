"""Utility classes for constructing inductive datasets.

These share the machinery of :mod:`pykeen.datasets.base` -- lazy loading, caching, summaries, and binary
(de)serialization -- and only differ in *which* splits they have and in how those splits share their entity and
relation indices, cf. ``pykeen.datasets.loaders.INDUCTIVE_PLAN``.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Self, cast

from pystow.utils.download import DownloadKwargs

from ..base import DatasetBase, LazyFactoryMixin
from ..loaders import INDUCTIVE_PLAN, PreSplitLoader
from ..sources import LocalSource, RemoteFile, RemoteSource, Source
from ...triples import CoreTriplesFactory, TriplesFactory

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


class InductiveDataset(DatasetBase):
    """Contains transductive train and inductive inference/validation/test datasets."""

    _factory_keys: ClassVar[tuple[str, ...]] = (
        "transductive_training",
        "inductive_inference",
        "inductive_testing",
        "inductive_validation",
    )

    #: A factory wrapping the training triples
    transductive_training: CoreTriplesFactory
    #: A factory wrapping the inductive inference triples that MIGHT or MIGHT NOT
    # share indices with the transductive training
    inductive_inference: CoreTriplesFactory
    #: A factory wrapping the testing triples, that share indices with the INDUCTIVE INFERENCE triples
    inductive_testing: CoreTriplesFactory
    #: A factory wrapping the validation triples, that share indices with the INDUCTIVE INFERENCE triples
    inductive_validation: CoreTriplesFactory | None = None
    #: All datasets should take care of inverse triple creation.
    #: note: unlike :class:`~pykeen.datasets.base.Dataset`, this is a plain attribute rather than being derived
    #: from the reference factory, since it is declared up-front by the dataset rather than by its files.
    create_inverse_triples: bool = True

    @classmethod
    def _eager_cls(cls) -> type[DatasetBase]:  # noqa: D102
        return EagerInductiveDataset

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
    inductive_validation: CoreTriplesFactory | None = None
    create_inverse_triples: bool = True
    metadata: Mapping[str, Any] | None = None


class LazyInductiveDataset(LazyFactoryMixin, InductiveDataset):
    """An inductive dataset that has lazy loading."""

    @property
    def transductive_training(self) -> TriplesFactory:  # type: ignore[override]  # noqa: D401
        """The training triples factory."""
        return cast(TriplesFactory, self.factory_dict["transductive_training"])

    @property
    def inductive_inference(self) -> TriplesFactory:  # type: ignore[override]  # noqa: D401
        """The inductive inference triples factory. MIGHT or MIGHT NOT share indices with the transductive train."""
        return cast(TriplesFactory, self.factory_dict["inductive_inference"])

    @property
    def inductive_testing(self) -> TriplesFactory:  # type: ignore[override]  # noqa: D401
        """The testing triples factory that share indices with the INDUCTIVE INFERENCE triples factory."""
        return cast(TriplesFactory, self.factory_dict["inductive_testing"])

    @property
    def inductive_validation(self) -> TriplesFactory | None:  # type: ignore[override]  # noqa: D401
        """The validation triples factory that shares indices with the INDUCTIVE INFERENCE triples factory."""
        return cast("TriplesFactory | None", self.factory_dict.get("inductive_validation"))


class DisjointInductivePathDataset(LazyInductiveDataset):
    """A disjoint inductive dataset specified by paths.

    Contains a lazy reference to a training, inductive inference, inductive testing, and inductive validation dataset.
    In this dataset, inductive inference is disjoint with the transductive train
    """

    def __init__(
        self,
        source: Source,
        *,
        eager: bool = False,
        create_inverse_triples: bool = False,
        load_triples_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the dataset.

        :param source: The source of the files, which provides the keys ``"transductive_training"``,
            ``"inductive_inference"``, ``"inductive_testing"``, and ``"inductive_validation"``. For local files, use
            :meth:`from_paths`.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param load_triples_kwargs: Arguments to pass through to :func:`~pykeen.triples.TriplesFactory.from_path`
            and ultimately through to :func:`~pykeen.triples.utils.load_triples`.
        """
        self.source = source
        self.create_inverse_triples = create_inverse_triples
        self.load_triples_kwargs = load_triples_kwargs
        super().__init__(
            loader=PreSplitLoader(
                source=source,
                plan=INDUCTIVE_PLAN,
                create_inverse_triples=create_inverse_triples,
                factory_cls=cast(type[TriplesFactory], self.triples_factory_cls),
                load_triples_kwargs=load_triples_kwargs,
            ),
            eager=eager,
        )

    @classmethod
    def from_paths(
        cls,
        transductive_training_path: str | pathlib.Path,
        inductive_inference_path: str | pathlib.Path,
        inductive_testing_path: str | pathlib.Path,
        inductive_validation_path: str | pathlib.Path,
        **kwargs: Any,
    ) -> Self:
        """Create a dataset from local files, one per split.

        .. note::

            This only works for subclasses which keep the base class' ``__init__`` signature.

        :param transductive_training_path: Path to the transductive training triples file.
        :param inductive_inference_path: Path to the inductive inference triples file.
        :param inductive_testing_path: Path to the inductive testing triples file.
        :param inductive_validation_path: Path to the inductive validation triples file.
        :param kwargs: Additional keyword-based parameters passed to ``__init__``.

        :returns: The dataset.
        """
        return cls(
            source=LocalSource(
                transductive_training=transductive_training_path,
                inductive_inference=inductive_inference_path,
                inductive_testing=inductive_testing_path,
                inductive_validation=inductive_validation_path,
            ),
            **kwargs,
        )

    def _path(self, key: str) -> pathlib.Path | None:
        return self.source.get_manifest().get(key)

    @property
    def transductive_training_path(self) -> pathlib.Path | None:
        """The path of the transductive training triples file."""
        return self._path("transductive_training")

    @property
    def inductive_inference_path(self) -> pathlib.Path | None:
        """The path of the inductive inference triples file."""
        return self._path("inductive_inference")

    @property
    def inductive_testing_path(self) -> pathlib.Path | None:
        """The path of the inductive testing triples file."""
        return self._path("inductive_testing")

    @property
    def inductive_validation_path(self) -> pathlib.Path | None:
        """The path of the inductive validation triples file."""
        return self._path("inductive_validation")

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
        cache_root: str | None = None,
        force: bool = False,
        eager: bool = False,
        create_inverse_triples: bool = False,
        load_triples_kwargs: Mapping[str, Any] | None = None,
        download_kwargs: DownloadKwargs | None = None,
        version: str | None = None,
    ):
        """Initialize dataset.

        :param transductive_training_url: The URL of the training file
        :param inductive_inference_url: The URL of the inductive inference graph file
        :param inductive_testing_url: The URL of the inductive testing file
        :param inductive_validation_url: The URL of the inductive validation file
        :param cache_root: An optional directory to store the extracted files. Is none is given, the default PyKEEN
            directory is used. This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to
            ``~/.data/pykeen``.
        :param force: If true, redownload any cached files
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param load_triples_kwargs: Arguments to pass through to :func:`~pykeen.triples.TriplesFactory.from_path`
            and ultimately through to :func:`~pykeen.triples.utils.load_triples`.
        :param download_kwargs: Keyword arguments to pass to :func:`pystow.utils.download`
        :param version: accepts a string "v1" to "v4" to select among Teru et al inductive datasets
        """
        self.version = version
        self.cache_root = self._help_cache(cache_root)

        self.transductive_training_url = transductive_training_url
        self.inductive_inference_url = inductive_inference_url
        self.inductive_testing_url = inductive_testing_url
        self.inductive_validation_url = inductive_validation_url

        super().__init__(
            source=RemoteSource(
                # note: the transductive training graph and the inductive part are kept in separate directories
                files=[
                    RemoteFile(key="transductive_training", url=transductive_training_url, sub_directory="training"),
                    RemoteFile(key="inductive_inference", url=inductive_inference_url, sub_directory="inference"),
                    RemoteFile(key="inductive_testing", url=inductive_testing_url, sub_directory="inference"),
                    RemoteFile(key="inductive_validation", url=inductive_validation_url, sub_directory="inference"),
                ],
                cache_root=self.cache_root,
                force=force,
                download_kwargs=download_kwargs,
            ),
            eager=eager,
            create_inverse_triples=create_inverse_triples,
            load_triples_kwargs=load_triples_kwargs,
        )

    def _cache_sub_directories(self) -> Iterable[str]:  # noqa: D102
        yield from super()._cache_sub_directories()
        # add v1 / v2 / v3 / v4 for inductive splits if available
        if self.version:
            yield self.version
