# -*- coding: utf-8 -*-

"""Implementation of factory that create instances containing of triples and numeric literals.tsv."""

from __future__ import annotations

import logging
import pathlib
import re
from re import Pattern
from typing import Any, ClassVar, Dict, Iterable, Mapping, MutableMapping, Optional, TextIO, Tuple, Union

import numpy as np
import pandas
import torch
from class_resolver import FunctionResolver, Hint, OptionalKwargs

from .triples_factory import Labeling, TriplesFactory
from .utils import load_triples
from ..typing import LabeledTriples, MappedTriples, NdArrayInOutCallable
from ..utils import format_relative_comparison, minmax_normalize

__all__ = [
    "TriplesNumericLiteralsFactory",
]


logger = logging.getLogger(__name__)


def create_matrix_of_literals(
    numeric_triples: np.ndarray,
    entity_labeling: Labeling,
    relation_regex: Union[Pattern, str, None] = None,
    min_occurrence: int = 0,
) -> Tuple[np.ndarray, Labeling]:
    """
    Create matrix of literals where each row corresponds to an entity and each column to a literal.

    :param numeric_triples: shape: (n, 3)
        the numeric triples, each a triple (entity, attribute_label, attribute_value)
    :param entity_labeling:
        the mapping from entity labels to IDs
    :param relation_regex:
        an optional filter-regex for attribute relations
    :param min_occurrence:
        a minimum number of occurrence to be considered

    :return:
        a pair (literal_matrix, attribute_relation_to_id), where `literal_matrix` is a matrix of shape
        `(num_entities, num_attribute_relations)`, and `attribute_relation_to_id` a mapping from attribute
        relation labels to their IDs.
    """
    entity_labels, attribute_relation_labels, attribute_values = numeric_triples.T
    # convert entity labels to IDs
    entity_ids = entity_labeling._vectorized_mapper(entity_labels, -1)
    triple_mask = entity_ids >= 0
    if not triple_mask.all():
        logger.warning(
            f"Dropping {format_relative_comparison(part=(~triple_mask).sum().item(), total=len(triple_mask))} "
            f"triples with invalid entity labels.",
        )
    # apply attribute relation filter
    uniq, inverse, counts = np.unique(attribute_relation_labels, return_counts=True, return_inverse=True)
    uniq_mask = np.ones_like(uniq, dtype=bool)
    if relation_regex:
        if isinstance(relation_regex, str):
            relation_regex = re.compile(relation_regex)
        uniq_mask &= np.asarray([bool(relation_regex.match(relation_label)) for relation_label in uniq.tolist()])
    if min_occurrence:
        uniq_mask &= counts >= min_occurrence
    triple_mask &= uniq_mask[inverse]
    logger.info(
        f"Keeping {format_relative_comparison(part=uniq_mask.sum().item(), total=len(uniq_mask))} attribute "
        f"relations. This leads to keeping "
        f"{format_relative_comparison(part=triple_mask.sum().item(), total=len(triple_mask))} of attribute triples.",
    )
    uniq = uniq[uniq_mask]
    # create mapping *after* filtering
    data_rel_labeling = Labeling(label_to_id={value: key for key, value in enumerate(sorted(uniq.tolist()))})
    # map
    attribute_relation_ids = data_rel_labeling._vectorized_mapper(attribute_relation_labels[triple_mask])
    # apply mask
    entity_ids = entity_ids[triple_mask]
    attribute_values = attribute_values[triple_mask]
    # create matrix
    literal_matrix = np.zeros([entity_labeling.max_id, data_rel_labeling.max_id], dtype=np.float32)
    # row define entity, and column the literal. Set the corresponding literal for the entity
    literal_matrix[entity_ids, attribute_relation_ids] = attribute_values
    return literal_matrix, data_rel_labeling


literal_matrix_preprocessing_resolver: FunctionResolver[NdArrayInOutCallable] = FunctionResolver(
    elements=[minmax_normalize], synonyms=dict(minmax=minmax_normalize)
)


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
        """Load relation triples and numeric attributive triples and call from_labeled_triples() for preprocessing.

        :param path: file path for relation triples
        :param path_to_numeric_triples:  file path for numeric attributive triples, defaults to None
        :param kwargs: Passed to the superclass

        :raises ValueError: if path_to_numeric_triples was not provided
        :return: an object of this class
        """
        if path_to_numeric_triples is None:
            raise ValueError(f"{cls.__name__} requires path_to_numeric_triples.")
        numeric_triples = load_triples(path_to_numeric_triples)
        triples = load_triples(path)
        return cls.from_labeled_triples(
            triples=triples,
            numeric_triples=numeric_triples,
            **kwargs,
        )

    # docstr-coverage: inherited
    @classmethod
    def from_labeled_triples(
        cls,
        triples: LabeledTriples,
        *,
        numeric_triples: Union[LabeledTriples, None] = None,
        relation_regex: Union[Pattern, str, None] = None,
        min_occurrence: int = 0,
        literal_matrix_preprocessing: Hint[NdArrayInOutCallable] = None,
        literal_matrix_preprocessing_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> "TriplesNumericLiteralsFactory":  # noqa: D102
        """Handle preprocessing of numeric attributive triples and their literals before creating an object of this class.

        :param triples: already loaded relation triples
        :param numeric_triples: already loaded numeric attributive triples, defaults to None
        :param relation_regex:
            a regular expression for attribute relations to keep. If `None`, keep all attribute relations.
        :param min_occurrence:
            a minimum occurrence count for an attribute relation to keep.
        :param literal_matrix_preprocessing:
            function to preprocess the numeric literals. Defaults to `None` which does not apply any operation.
            cf. `num_literals_preproc_resolver` for further options.
        :param literal_matrix_preprocessing_kwargs:
            keyword-based parameters passed to the literal matrix preprocessing function.
        :param kwargs:
            passed to :meth:`TriplesFactory.from_labeled_triples`

        :raises ValueError: if numeric_triples was not provided
        :return: an object of this class
        """
        if numeric_triples is None:
            raise ValueError(f"{cls.__name__} requires numeric_triples.")
        base = TriplesFactory.from_labeled_triples(triples=triples, **kwargs)
        numeric_literals, literals_to_id = create_matrix_of_literals(
            numeric_triples=numeric_triples,
            entity_labeling=base.entity_labeling,
            relation_regex=relation_regex,
            min_occurrence=min_occurrence,
        )
        # apply optional preprocessing
        if literal_matrix_preprocessing is not None:
            preprocessing_function = literal_matrix_preprocessing_resolver.make(
                query=literal_matrix_preprocessing, pos_kwargs=literal_matrix_preprocessing_kwargs
            )
            numeric_literals = preprocessing_function(numeric_literals)
        return cls(
            entity_to_id=base.entity_to_id,
            relation_to_id=base.relation_to_id,
            mapped_triples=base.mapped_triples,
            create_inverse_triples=base.create_inverse_triples,
            numeric_literals=numeric_literals,
            literals_to_id=literals_to_id.label_to_id,
        )

    def get_numeric_literals_tensor(self) -> torch.FloatTensor:
        """Return the numeric literals as a tensor."""
        return torch.as_tensor(self.numeric_literals, dtype=torch.get_default_dtype())

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
