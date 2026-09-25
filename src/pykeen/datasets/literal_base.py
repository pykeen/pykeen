"""Base classes for literal datasets."""

import pathlib

from .base import PathDataset
from .sources import Source
from ..triples import TriplesNumericLiteralsFactory

__all__ = [
    "NumericPathDataset",
]


class NumericPathDataset(PathDataset):
    """A path dataset which additionally loads numeric literals for its entities.

    This is an ordinary :class:`~pykeen.datasets.base.PathDataset` with a different triples factory class; the
    literals file is passed to every factory.
    """

    triples_factory_cls = TriplesNumericLiteralsFactory

    def __init__(
        self,
        source: Source,
        literals_path: str | pathlib.Path,
        *,
        eager: bool = False,
        create_inverse_triples: bool = False,
    ) -> None:
        """Initialize the dataset.

        :param source: The source of the triples files, cf. :class:`~pykeen.datasets.base.PathDataset`. For local
            files, use :meth:`~pykeen.datasets.base.PathDataset.from_paths` with an additional ``literals_path``.
        :param literals_path: Path to the literals triples file.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        """
        self.literals_path = pathlib.Path(literals_path)
        super().__init__(
            source=source,
            eager=eager,
            create_inverse_triples=create_inverse_triples,
            factory_kwargs={"path_to_numeric_triples": self.literals_path},
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
