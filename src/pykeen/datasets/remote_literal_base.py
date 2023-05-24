# -*- coding: utf-8 -*-

"""Base classes for remote literal datasets."""

from __future__ import annotations

import logging
import pathlib
from re import Pattern
import zipfile
from typing import Any, Mapping, Optional, Tuple, Union


import pandas as pd
from class_resolver import Hint, OptionalKwargs
from pystow.utils import download, name_from_url

from .base import PackedZipRemoteDataset, TarFileRemoteDataset
from ..triples import TriplesFactory, TriplesNumericLiteralsFactory
from ..triples.utils import load_triples
from ..typing import NdArrayInOutCallable

logger = logging.getLogger(__name__)


def get_literal_summary(triples_factory: TriplesNumericLiteralsFactory) -> Tuple[str, str, int, int]:
    """Construct a tuple with information about numeric literals.

    :param triples_factory: triples factory including numeric attributive triples
    """
    assert isinstance(triples_factory, TriplesNumericLiteralsFactory)
    n_relations = len(triples_factory.literals_to_id)
    n_triples = n_relations * triples_factory.num_entities

    return ("Literals", "-", n_relations, n_triples)


class ZipRemoteDatasetWithRemoteLiterals(PackedZipRemoteDataset):
    """Extends remote zip dataset with literals from txt file."""

    triples_factory_cls = TriplesNumericLiteralsFactory

    def __init__(
        self,
        numeric_triples_url: str,
        relation_regex: Union[Pattern, str, None] = None,
        min_occurrence: int = 0,
        literal_matrix_preprocessing: Hint[NdArrayInOutCallable] = None,
        literal_matrix_preprocessing_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        """Initialize fields regarding numeric attributive triples and let the parent class handle the rest of the args.

        :param numeric_triples_url: URL of the text file with the numeric attributive triples
        :param literal_matrix_preprocessing: function for preprocessing numeric literals, defaults to None
               e.g. ..utils.minmax_normalize() can be used or a custom function to modify literals
        :param literal_matrix_preprocessing_kwargs: args to pass to the above preprocessing function, defaults to None
        :param kwargs: Passed to the superclass
        """
        super().__init__(**kwargs)

        self.numeric_triples_url = numeric_triples_url
        self.relation_regex = relation_regex
        self.min_occurrence = min_occurrence
        self.literal_matrix_preprocessing = literal_matrix_preprocessing
        self.literal_matrix_preprocessing_kwargs = literal_matrix_preprocessing_kwargs

        self.numeric_triples_file_name = name_from_url(self.numeric_triples_url)
        self.path_to_numeric_triples = self.cache_root.joinpath(self.numeric_triples_file_name)

    def _load_helper(
        self,
        relative_path: pathlib.PurePath,
        entity_to_id: Optional[Mapping[str, Any]] = None,
        relation_to_id: Optional[Mapping[str, Any]] = None,
    ) -> TriplesFactory:
        """Load relation triples from remote zip file and numeric attributive triples from remote text file."""
        if not self.path.is_file():
            if self.url is None:
                raise ValueError("url should be set")
            logger.info("downloading data from %s to %s", self.url, self.path)
            download(url=self.url, path=self.path)
            download(url=self.numeric_triples_url, path=self.path_to_numeric_triples)

        with zipfile.ZipFile(file=self.path) as zf:
            # relative paths within zip file's always follow Posix path, even on Windows
            with zf.open(relative_path.as_posix()) as file:
                logger.debug("loading %s", relative_path)
                df = pd.read_csv(
                    file,
                    usecols=[self.head_column, self.relation_column, self.tail_column],
                    header=self.header,
                    sep=self.sep,
                )
                numeric_triples = load_triples(self.path_to_numeric_triples)
                return self.triples_factory_cls.from_labeled_triples(
                    triples=df.values,
                    create_inverse_triples=self._create_inverse_triples,
                    metadata={"path": relative_path},
                    entity_to_id=entity_to_id,
                    relation_to_id=relation_to_id,
                    numeric_triples=numeric_triples,
                    relation_regex=self.relation_regex,
                    min_occurrence=self.min_occurrence,
                    literal_matrix_preprocessing=self.literal_matrix_preprocessing,
                    literal_matrix_preprocessing_kwargs=self.literal_matrix_preprocessing_kwargs,
                )

    def _summary_rows(self):
        """Enhance dataset's summary with information about numeric literals.

        :return: enhanced summary
        """
        rv = super()._summary_rows()
        tf = self.training
        rv.append(get_literal_summary(tf))
        return rv


class TarRemoteDatasetWithRemoteLiterals(TarFileRemoteDataset):
    """Extends remote tar dataset with literals from txt file."""

    triples_factory_cls = TriplesNumericLiteralsFactory

    def __init__(
        self,
        numeric_triples_url: str,
        relation_regex: Union[Pattern, str, None] = None,
        min_occurrence: int = 0,
        literal_matrix_preprocessing: Hint[NdArrayInOutCallable] = None,
        literal_matrix_preprocessing_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        """Initialize fields regarding numeric attributive triples and lets the parent class handle the rest of the args.

        :param numeric_triples_preprocessing_kwargs: args to pass to the above preprocessing function, defaults to None
        :param literal_matrix_preprocessing: function for preprocessing numeric literals, defaults to None
               e.g. ..utils.minmax_normalize() can be used or a custom function to modify literals
        :param literal_matrix_preprocessing_kwargs: args to pass to the above preprocessing function, defaults to None
        :param kwargs: Passed to the superclass
        """
        super().__init__(**kwargs)

        self.numeric_triples_url = numeric_triples_url
        self.relation_regex = relation_regex
        self.min_occurrence = min_occurrence
        self.literal_matrix_preprocessing = literal_matrix_preprocessing
        self.literal_matrix_preprocessing_kwargs = literal_matrix_preprocessing_kwargs

        self.numeric_triples_file_name = name_from_url(self.numeric_triples_url)
        self.path_to_numeric_triples = self.cache_root.joinpath(self.numeric_triples_file_name)  # noqa

    # docstr-coverage: inherited
    def _load(self) -> None:  # noqa: D102
        """Load train & test relation triples from remote zip file and numeric attr. triples from remote text file."""
        all_unpacked = all(path.is_file() for path in self._get_paths())

        if not all_unpacked:
            archive_file = self._get_bytes()
            self._extract(archive_file=archive_file)
            logger.info(f"Extracted to {self.cache_root}.")

        download(url=self.numeric_triples_url, path=self.path_to_numeric_triples)

        self._training = self.triples_factory_cls.from_path(
            path=self.training_path,
            create_inverse_triples=self._create_inverse_triples,
            path_to_numeric_triples=self.path_to_numeric_triples,
            relation_regex=self.relation_regex,
            min_occurrence=self.min_occurrence,
            literal_matrix_preprocessing=self.literal_matrix_preprocessing,
            literal_matrix_preprocessing_kwargs=self.literal_matrix_preprocessing_kwargs,
        )
        self._testing = self.triples_factory_cls.from_path(
            path=self.testing_path,
            entity_to_id=self._training.entity_to_id,  # share entity index with training
            relation_to_id=self._training.relation_to_id,  # share relation index with training
            create_inverse_triples=self._create_inverse_triples,
            path_to_numeric_triples=self.path_to_numeric_triples,
            relation_regex=self.relation_regex,
            min_occurrence=self.min_occurrence,
            literal_matrix_preprocessing=self.literal_matrix_preprocessing,
            literal_matrix_preprocessing_kwargs=self.literal_matrix_preprocessing_kwargs,  # noqa
        )

    def _load_validation(self) -> None:
        """Load validation relation triples from remote zip file and numeric attr. triples from remote text file."""
        # don't call this function by itself. assumes called through the `validation`
        # property and the _training factory has already been loaded
        assert self._training is not None
        if self.validation_path is None:
            self._validation = None
        else:
            self._validation = self.triples_factory_cls.from_path(
                path=self.validation_path,
                entity_to_id=self._training.entity_to_id,  # share entity index with training
                relation_to_id=self._training.relation_to_id,  # share relation index with training
                create_inverse_triples=self._create_inverse_triples,
                path_to_numeric_triples=self.path_to_numeric_triples,
                relation_regex=self.relation_regex,
                min_occurrence=self.min_occurrence,
                literal_matrix_preprocessing=self.literal_matrix_preprocessing,
                literal_matrix_preprocessing_kwargs=self.literal_matrix_preprocessing_kwargs,
            )

    def _summary_rows(self):
        """Enhance dataset's summary with information about numeric literals.

        :return: enhanced summary
        """
        rv = super()._summary_rows()
        tf = self.training
        rv.append(get_literal_summary(tf))
        return rv
