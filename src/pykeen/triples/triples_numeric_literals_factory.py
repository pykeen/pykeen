# -*- coding: utf-8 -*-

"""Implementation of factory that create instances containing of triples and numeric literals.tsv."""

import logging
import pathlib
from typing import Any, ClassVar, Dict, Iterable, Mapping, MutableMapping, Optional, TextIO, Tuple, Union

import numpy as np
import pandas
import torch

from .triples_factory import TriplesFactory
from .utils import load_triples
from ..typing import EntityMapping, LabeledTriples, MappedTriples

__all__ = [
    "TriplesNumericLiteralsFactory",
]

logger = logging.getLogger(__name__)


def create_matrix_of_literals(
    numeric_triples: np.array,
    entity_to_id: EntityMapping,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Create matrix of literals where each row corresponds to an entity and each column to a literal."""
    data_relations = np.unique(np.ndarray.flatten(numeric_triples[:, 1:2]))
    data_rel_to_id: Dict[str, int] = {value: key for key, value in enumerate(data_relations)}
    # Prepare literal matrix, set every literal to zero, and afterwards fill in the corresponding value if available
    num_literals = np.zeros([len(entity_to_id), len(data_rel_to_id)], dtype=np.float32)

    # TODO vectorize code
    for h, r, lit in numeric_triples:
        try:
            # row define entity, and column the literal. Set the corresponding literal for the entity
            num_literals[entity_to_id[h], data_rel_to_id[r]] = lit
        except KeyError:
            logger.info("Either entity or relation to literal doesn't exist.")
            continue

    return num_literals, data_rel_to_id


class TriplesNumericLiteralsFactory(TriplesFactory):
    """Create multi-modal instances given the path to triples."""

    file_name_literal_to_id: ClassVar[str] = "literal_to_id"
    file_name_numeric_literals: ClassVar[str] = "literals"

    def __init__(
        self,
        *,
        numeric_literals: np.ndarray,
        literals_to_id: Mapping[str, int],
        **kwargs,
    ) -> None:
        """Initialize the multi-modal triples factory.

        :param numeric_literals: shape: (num_entities, num_literals)
            the numeric literals as a dense matrix.
        :param literals_to_id:
            a mapping from literal names to their IDs, i.e., the columns in the `numeric_literals` matrix.
        :param kwargs:
            additional keyword-based parameters passed to :meth:`TriplesFactory.__init__`.
        """
        super().__init__(**kwargs)
        self.numeric_literals = numeric_literals
        self.literals_to_id = literals_to_id

    # docstr-coverage: inherited
    @classmethod
    def from_path(
        cls,
        path: Union[str, pathlib.Path, TextIO],
        *,
        path_to_numeric_triples: Union[None, str, pathlib.Path, TextIO] = None,
        **kwargs,
    ) -> "TriplesNumericLiteralsFactory":  # noqa: D102
        if path_to_numeric_triples is None:
            raise ValueError(f"{cls.__name__} requires path_to_numeric_triples.")
        numeric_triples = load_triples(path_to_numeric_triples)
        triples = load_triples(path)
        return cls.from_labeled_triples(triples=triples, numeric_triples=numeric_triples, **kwargs)

    # docstr-coverage: inherited
    @classmethod
    def from_labeled_triples(
        cls,
        triples: LabeledTriples,
        *,
        numeric_triples: LabeledTriples = None,
        **kwargs,
    ) -> "TriplesNumericLiteralsFactory":  # noqa: D102
        if numeric_triples is None:
            raise ValueError(f"{cls.__name__} requires numeric_triples.")
        base = TriplesFactory.from_labeled_triples(triples=triples, **kwargs)
        numeric_literals, literals_to_id = create_matrix_of_literals(
            numeric_triples=numeric_triples, entity_to_id=base.entity_to_id
        )
        return cls(
            entity_to_id=base.entity_to_id,
            relation_to_id=base.relation_to_id,
            mapped_triples=base.mapped_triples,
            create_inverse_triples=base.create_inverse_triples,
            numeric_literals=numeric_literals,
            literals_to_id=literals_to_id,
        )

    def get_numeric_literals_tensor(self) -> torch.FloatTensor:
        """Return the numeric literals as a tensor."""
        return torch.as_tensor(self.numeric_literals, dtype=torch.float32)

    @property
    def literal_shape(self) -> Tuple[int, ...]:
        """Return the shape of the literals."""
        return self.numeric_literals.shape[1:]

    # docstr-coverage: inherited
    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"num_literals={len(self.literals_to_id)}"

    # docstr-coverage: inherited
    def clone_and_exchange_triples(
        self,
        mapped_triples: MappedTriples,
        extra_metadata: Optional[Dict[str, Any]] = None,
        keep_metadata: bool = True,
        create_inverse_triples: Optional[bool] = None,
    ) -> "TriplesNumericLiteralsFactory":  # noqa: D102
        if create_inverse_triples is None:
            create_inverse_triples = self.create_inverse_triples
        return TriplesNumericLiteralsFactory(
            mapped_triples=mapped_triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            create_inverse_triples=create_inverse_triples,
            metadata={
                **(extra_metadata or {}),
                **(self.metadata if keep_metadata else {}),  # type: ignore
            },
            numeric_literals=self.numeric_literals,
            literals_to_id=self.literals_to_id,
        )

    # docstr-coverage: inherited
    def to_path_binary(self, path: Union[str, pathlib.Path, TextIO]) -> pathlib.Path:  # noqa: D102
        path = super().to_path_binary(path=path)
        # save literal-to-id mapping
        pandas.DataFrame(
            data=self.literals_to_id.items(),
            columns=["label", "id"],
        ).sort_values(
            by="id"
        ).set_index("id").to_csv(
            path.joinpath(f"{self.file_name_literal_to_id}.tsv.gz"),
            sep="\t",
        )
        # save numeric literals
        np.save(str(path.joinpath(self.file_name_numeric_literals)), self.numeric_literals)
        return path

    @classmethod
    def _from_path_binary(cls, path: pathlib.Path) -> MutableMapping[str, Any]:
        data = super()._from_path_binary(path)
        # load literal-to-id
        df = pandas.read_csv(
            path.joinpath(f"{cls.file_name_literal_to_id}.tsv.gz"),
            sep="\t",
        )
        data["literals_to_id"] = dict(zip(df["label"], df["id"]))
        # load literals
        data["numeric_literals"] = np.load(
            str(path.joinpath(cls.file_name_numeric_literals).with_suffix(suffix=".npy"))
        )
        return data
