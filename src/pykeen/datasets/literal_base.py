# -*- coding: utf-8 -*-

"""Base classes for literal datasets."""

import pathlib
from typing import TextIO, Union

from .base import LazyDataset
from ..triples import TriplesNumericLiteralsFactory

__all__ = [
    "NumericPathDataset",
]


class NumericPathDataset(LazyDataset):
    """Contains a lazy reference to a training, testing, and validation dataset."""

    triples_factory_cls = TriplesNumericLiteralsFactory

    def __init__(
        self,
        training_path: Union[str, pathlib.Path, TextIO],
        testing_path: Union[str, pathlib.Path, TextIO],
        validation_path: Union[str, pathlib.Path, TextIO],
        literals_path: Union[str, pathlib.Path, TextIO],
        eager: bool = False,
        create_inverse_triples: bool = False,
    ) -> None:
        """Initialize the dataset.

        :param training_path: Path to the training triples file or training triples file.
        :param testing_path: Path to the testing triples file or testing triples file.
        :param validation_path: Path to the validation triples file or validation triples file.
        :param literals_path: Path to the literals triples file or literal triples file
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        """
        self.training_path = training_path
        self.testing_path = testing_path
        self.validation_path = validation_path
        self.literals_path = literals_path

        self._create_inverse_triples = create_inverse_triples

        if eager:
            self._load()
            self._load_validation()

    def _load(self) -> None:
        self._training = self.triples_factory_cls.from_path(
            path=self.training_path,
            path_to_numeric_triples=self.literals_path,
            create_inverse_triples=self._create_inverse_triples,
        )
        self._testing = self.triples_factory_cls.from_path(
            path=self.testing_path,
            path_to_numeric_triples=self.literals_path,
            entity_to_id=self._training.entity_to_id,  # share entity index with training
            relation_to_id=self._training.relation_to_id,  # share relation index with training
        )

    def _load_validation(self) -> None:
        # don't call this function by itself. assumes called through the `validation`
        # property and the _training factory has already been loaded
        assert self._training is not None
        self._validation = self.triples_factory_cls.from_path(
            path=self.validation_path,
            path_to_numeric_triples=self.literals_path,
            entity_to_id=self._training.entity_to_id,  # share entity index with training
            relation_to_id=self._training.relation_to_id,  # share relation index with training
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f'{self.__class__.__name__}(training_path="{self.training_path}", testing_path="{self.testing_path}",'
            f' validation_path="{self.validation_path}", literals_path="{self.literals_path}")'
        )

    def _summary_rows(self):
        rv = super()._summary_rows()
        tf = self.training
        assert isinstance(tf, TriplesNumericLiteralsFactory)
        n_relations = len(tf.literals_to_id)
        n_triples = n_relations * tf.num_entities
        rv.append(("Literals", "-", n_relations, n_triples))
        return rv
