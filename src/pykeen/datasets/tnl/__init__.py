# -*- coding: utf-8 -*-

"""Test Nations Literal Dataset.

.. warning:: DO NOT USE THIS DATASET FOR BENCHMARKING. It is fake.
"""

import os
from typing import TextIO, Union

from pykeen.datasets.base import LazyDataset
from pykeen.datasets.nations import NATIONS_TEST_PATH, NATIONS_TRAIN_PATH, NATIONS_VALIDATE_PATH
from pykeen.triples import TriplesNumericLiteralsFactory

HERE = os.path.abspath(os.path.dirname(__file__))
LITERALS_PATH = os.path.join(HERE, 'literals.txt')


class NumericPathDataset(LazyDataset):
    """Contains a lazy reference to a training, testing, and validation dataset."""

    def __init__(
        self,
        training_path: Union[str, TextIO],
        testing_path: Union[str, TextIO],
        validation_path: Union[str, TextIO],
        literals_path: Union[str, TextIO],
        eager: bool = False,
        create_inverse_triples: bool = False,
    ) -> None:
        """Initialize the dataset.

        :param training_path: Path to the training triples file or training triples file.
        :param testing_path: Path to the testing triples file or testing triples file.
        :param validation_path: Path to the validation triples file or validation triples file.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        """
        self.training_path = training_path
        self.testing_path = testing_path
        self.validation_path = validation_path
        self.literals_path = literals_path

        self.create_inverse_triples = create_inverse_triples

        if eager:
            self._load()
            self._load_validation()

    def _load(self) -> None:
        self._training = TriplesNumericLiteralsFactory(
            path=self.training_path,
            path_to_numeric_triples=self.literals_path,
            create_inverse_triples=self.create_inverse_triples,
        )
        self._testing = TriplesNumericLiteralsFactory(
            path=self.testing_path,
            path_to_numeric_triples=self.literals_path,
            entity_to_id=self._training.entity_to_id,  # share entity index with training
            relation_to_id=self._training.relation_to_id,  # share relation index with training
        )

    def _load_validation(self) -> None:
        # don't call this function by itself. assumes called through the `validation`
        # property and the _training factory has already been loaded
        self._validation = TriplesNumericLiteralsFactory(
            path=self.validation_path,
            path_to_numeric_triples=self.literals_path,
            entity_to_id=self._training.entity_to_id,  # share entity index with training
            relation_to_id=self._training.relation_to_id,  # share relation index with training
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f'{self.__class__.__name__}(training_path="{self.training_path}", testing_path="{self.testing_path}",'
            f' validation_path="{self.validation_path}")'
        )

    def _summary_rows(self):
        rv = super()._summary_rows()
        n_relations = len(self.training.literals_to_id)
        n_triples = n_relations * self.training.num_entities
        rv.append(('Literals', '-', n_relations, n_triples))
        return rv


class NationsLiteralDataset(NumericPathDataset):
    """The Nations dataset with literals."""

    def __init__(self, **kwargs):
        super().__init__(
            training_path=NATIONS_TRAIN_PATH,
            testing_path=NATIONS_TEST_PATH,
            validation_path=NATIONS_VALIDATE_PATH,
            literals_path=LITERALS_PATH,
            **kwargs,
        )


def _main():
    NationsLiteralDataset().summarize()


if __name__ == '__main__':
    _main()
