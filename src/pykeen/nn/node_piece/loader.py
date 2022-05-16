# -*- coding: utf-8 -*-

"""Loaders for pre-computed NodePiece tokenizations."""

import logging
import pathlib
import pickle
from abc import ABC, abstractmethod
from typing import Collection, Mapping, Tuple

import numpy
import torch
from class_resolver import ClassResolver
from tqdm.auto import tqdm

__all__ = [
    # Resolver
    "precomputed_tokenizer_loader_resolver",
    # Base classes
    "PrecomputedTokenizerLoader",
    # Concrete classes
    "GalkinPrecomputedTokenizerLoader",
    "TorchPrecomputedTokenizerLoader",
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

    # docstr-coverage: inherited
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


class TorchPrecomputedTokenizerLoader(PrecomputedTokenizerLoader):
    """A loader via torch.load."""

    @staticmethod
    def save(path: pathlib.Path, order: numpy.ndarray, anchor_ids: numpy.ndarray) -> None:
        """
        Save tokenization to path.

        :param path:
            the output path
        :param order: shape: (num_entities, num_anchors)
            the sorted `anchor_ids`' ids per entity
        :param anchor_ids: shape: (num_anchors,)
            the anchor entity IDs
        """
        # ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        # save via torch.save
        torch.save(
            {
                "order": order,
                "anchors": anchor_ids,  # ignored for now
            },
            path,
        )

    # docstr-coverage: inherited
    def __call__(self, path: pathlib.Path) -> Tuple[Mapping[int, Collection[int]], int]:  # noqa: D102
        c = torch.load(path)
        order = c["order"]
        logger.info(f"Loaded precomputed pools of shape {order.shape}.")
        num_anchors = c["anchors"].shape[0]
        # TODO: since we save a contiguous array of (num_entities, num_anchors),
        # it would be more efficient to not convert to a mapping, but directly select from the tensor
        return {i: anchor_ids.tolist() for i, anchor_ids in enumerate(order)}, num_anchors  # type: ignore


precomputed_tokenizer_loader_resolver: ClassResolver[PrecomputedTokenizerLoader] = ClassResolver.from_subclasses(
    base=PrecomputedTokenizerLoader,
    default=GalkinPrecomputedTokenizerLoader,
)
