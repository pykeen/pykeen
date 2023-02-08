import logging
import pathlib
import zipfile
from typing import Callable, Optional, Mapping, Any

import numpy as np
import pandas as pd
from pystow.utils import name_from_url, download

from pykeen.datasets import PackedZipRemoteDataset, TarFileRemoteDataset
from pykeen.triples import TriplesNumericLiteralsFactory, TriplesFactory
from pykeen.triples.utils import load_triples

logger = logging.getLogger(__name__)


class ZipRemoteDatasetWithRemoteLiterals(PackedZipRemoteDataset):
    triples_factory_cls = TriplesNumericLiteralsFactory

    def __init__(
        self,
        numeric_triples_url: str,
        numeric_literals_preprocessing: Callable[[np.ndarray], np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.numeric_triples_url = numeric_triples_url
        self.numeric_literals_preprocessing = numeric_literals_preprocessing

        self.numeric_triples_file_name = name_from_url(self.numeric_triples_url)
        self.path_to_numeric_triples = self.cache_root.joinpath(self.numeric_triples_file_name)

    def _load_helper(
        self,
        relative_path: pathlib.PurePath,
        entity_to_id: Optional[Mapping[str, Any]] = None,
        relation_to_id: Optional[Mapping[str, Any]] = None,
    ) -> TriplesFactory:
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
                    numeric_literals_preprocessing=self.numeric_literals_preprocessing,
                )

    def _summary_rows(self):
        rv = super()._summary_rows()
        tf = self.training
        assert isinstance(tf, TriplesNumericLiteralsFactory)
        n_relations = len(tf.literals_to_id)
        n_triples = n_relations * tf.num_entities
        rv.append(("Literals", "-", n_relations, n_triples))
        return rv


class TarRemoteDatasetWithRemoteLiterals(TarFileRemoteDataset):
    triples_factory_cls = TriplesNumericLiteralsFactory

    def __init__(
        self,
        numeric_triples_url: str,
        numeric_literals_preprocessing: Callable[[np.ndarray], np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.numeric_triples_url = numeric_triples_url
        self.numeric_literals_preprocessing = numeric_literals_preprocessing

        self.numeric_triples_file_name = name_from_url(self.numeric_triples_url)
        self.path_to_numeric_triples = self.cache_root.joinpath(self.numeric_triples_file_name)

    # docstr-coverage: inherited
    def _load(self) -> None:  # noqa: D102
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
            numeric_literals_preprocessing=self.numeric_literals_preprocessing,
        )
        self._testing = self.triples_factory_cls.from_path(
            path=self.testing_path,
            entity_to_id=self._training.entity_to_id,  # share entity index with training
            relation_to_id=self._training.relation_to_id,  # share relation index with training
            create_inverse_triples=self._create_inverse_triples,
            path_to_numeric_triples=self.path_to_numeric_triples,
            numeric_literals_preprocessing=self.numeric_literals_preprocessing,
        )

    def _load_validation(self) -> None:
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
                numeric_literals_preprocessing=self.numeric_literals_preprocessing,
            )
