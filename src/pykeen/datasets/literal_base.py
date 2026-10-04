"""Base classes for literal datasets."""

import pathlib
from typing import IO, Unpack

from .base import LazyDataset, LazyDatasetKwargs
from ..triples import TriplesNumericLiteralsFactory

__all__ = [
    "NumericPathDataset",
]


class NumericPathDataset(LazyDataset):
    """Contains a lazy reference to a training, testing, and validation dataset."""

    triples_factory_cls = TriplesNumericLiteralsFactory

    def __init__(
        self,
        training_path: str | pathlib.Path | IO[str],
        testing_path: str | pathlib.Path | IO[str],
        validation_path: str | pathlib.Path | IO[str],
        literals_path: str | pathlib.Path | IO[str],
        **kwargs: Unpack[LazyDatasetKwargs],
    ) -> None:
        """Initialize the dataset.

        :param training_path: Path to the training triples file or training triples file.
        :param testing_path: Path to the testing triples file or testing triples file.
        :param validation_path: Path to the validation triples file or validation triples file.
        :param literals_path: Path to the literals triples file or literal triples file
        """
        self.training_path = training_path
        self.testing_path = testing_path
        self.validation_path = validation_path
        self.literals_path = literals_path
        super().__init__(**kwargs)

    def _load(self) -> None:
        self._training = self.triples_factory_cls.from_path(
            path=self.training_path,
            path_to_numeric_triples=self.literals_path,
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
        if self._training is None:
            raise RuntimeError("The training factory has to be loaded before the validation factory.")
        self._validation = self.triples_factory_cls.from_path(
            path=self.validation_path,
            path_to_numeric_triples=self.literals_path,
            entity_to_id=self._training.entity_to_id,  # share entity index with training
            relation_to_id=self._training.relation_to_id,  # share relation index with training
        )

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(training_path="{self.training_path}", testing_path="{self.testing_path}",'
            f' validation_path="{self.validation_path}", literals_path="{self.literals_path}")'
        )

    def _summary_rows(self):
        rv = super()._summary_rows()
        tf = self.training
        n_relations = len(tf.literals_to_id)
        n_triples = n_relations * tf.num_entities
        rv.append(("Literals", "-", n_relations, n_triples))
        return rv
