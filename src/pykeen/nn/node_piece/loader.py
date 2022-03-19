# -*- coding: utf-8 -*-

"""Loaders for pre-computed NodePiece tokenizations."""

import logging
import pathlib
import pickle
from abc import ABC, abstractmethod
from typing import Collection, Mapping, Tuple

from class_resolver import ClassResolver
from tqdm.auto import tqdm

__all__ = [
    # Resolver
    "precomputed_tokenizer_loader_resolver",
    # Base classes
    "PrecomputedTokenizerLoader",
    # Concrete classes
    "GalkinPrecomputedTokenizerLoader",
]

logger = logging.getLogger(__name__)


class PrecomputedTokenizerLoader(ABC):
    """A loader for precomputed tokenization."""

    @abstractmethod
    def __call__(self, path: pathlib.Path) -> Tuple[Mapping[int, Collection[int]], int]:
        """Load tokenization from the given path."""
        raise NotImplementedError


class GalkinPrecomputedTokenizerLoader(PrecomputedTokenizerLoader):
    """
    A loader for pickle files provided by Galkin *et al*.

    .. seealso ::
        https://github.com/migalkin/NodePiece/blob/9adc57efe302919d017d74fc648f853308cf75fd/download_data.sh
        https://github.com/migalkin/NodePiece/blob/9adc57efe302919d017d74fc648f853308cf75fd/ogb/download.sh
    """

    def __call__(self, path: pathlib.Path) -> Tuple[Mapping[int, Collection[int]], int]:  # noqa: D102
        with path.open(mode="rb") as pickle_file:
            # contains: anchor_ids, entity_ids, mapping {entity_id -> {"ancs": anchors, "dists": distances}}
            anchor_ids, mapping = pickle.load(pickle_file)[0::2]
        logger.info(f"Loaded precomputed pools with {len(anchor_ids)} anchors, and {len(mapping)} pools.")
        # normalize anchor_ids
        anchor_map = {a: i for i, a in enumerate(anchor_ids) if a >= 0}
        # cf. https://github.com/pykeen/pykeen/pull/822#discussion_r822889541
        # TODO: keep distances?
        return {
            key: [anchor_map[a] for a in value["ancs"] if a in anchor_map]
            for key, value in tqdm(mapping.items(), desc="ID Mapping", unit_scale=True, leave=False)
        }, len(anchor_map)


precomputed_tokenizer_loader_resolver: ClassResolver[PrecomputedTokenizerLoader] = ClassResolver.from_subclasses(
    base=PrecomputedTokenizerLoader,
    default=GalkinPrecomputedTokenizerLoader,
)
