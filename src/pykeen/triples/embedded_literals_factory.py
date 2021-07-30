# -*- coding: utf-8 -*-

"""Implementation of factory that create instances containing of triples and textual embedded literals."""

__all__ = [
    'EmbeddedLiteralsFactory',
]

import pathlib
from typing import Union, TextIO, Optional, Mapping

import numpy as np
from pykeen.triples import TriplesFactory
from pykeen.typing import LabeledTriples, EntityMapping


def get_remapping_index(
    initial_label_to_id: Mapping[str, int],
    new_label_to_id: Mapping[str, int],
) -> np.ndarray:
    """
    Create a remapping index based on two label-to-ID mappings for the same labels.

    :param initial_label_to_id:
        The initial mapping which is expected to be a superset of new_label_to_id.
    :param new_label_to_id:
        The new mapping.

    :return: shape: (n,)
        An index to remap embeddings from the old IDs to the new IDs by embeddings[index].
    """

    num = len(new_label_to_id)
    if set(new_label_to_id.values()) != set(range(num)):
        raise ValueError("The IDs are not consecutive.")

    result = np.empty(shape=(num,), dtype=np.int32)
    for label, new_id in new_label_to_id.items():
        result[new_id] = initial_label_to_id[label]
    return result


class EmbeddedLiteralsFactory(TriplesFactory):
    """Create multi-modal instances given the path to triples."""

    def __init__(
        self,
        *,
        path: Union[None, str, pathlib.Path, TextIO] = None,
        triples: Optional[LabeledTriples] = None,
        node_embeddings: np.ndarray,
        embedding_label_to_id: EntityMapping,
        **kwargs,
    ) -> None:
        """Initialize the multi-modal triples factory.

        :param path: The path to a 3-column TSV file with triples in it. If not specified,
         you should specify ``triples``.
        :param triples:  A 3-column numpy array with triples in it. If not specified,
         you should specify ``path``
        """
        if path is None:
            base = TriplesFactory.from_labeled_triples(triples=triples, **kwargs)
        else:
            base = TriplesFactory.from_path(path=path, **kwargs)
        super().__init__(
            entity_to_id=base.entity_to_id,
            relation_to_id=base.relation_to_id,
            mapped_triples=base.mapped_triples,
            create_inverse_triples=base.create_inverse_triples,
        )

        assert node_embeddings is not None
        assert self.entity_to_id is not None

        # Re-order embedding matrix based on entity_to_id
        self.node_embeddings = node_embeddings[
            get_remapping_index(initial_label_to_id=embedding_label_to_id, new_label_to_id=self.entity_to_id)
        ]
