# -*- coding: utf-8 -*-

"""Utilities for neural network components."""

from __future__ import annotations

import functools
import logging
import pathlib
import re
import subprocess
from abc import ABC, abstractmethod
from itertools import chain
from textwrap import dedent
from typing import Any, Callable, Collection, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Union, cast

import more_itertools
import requests
import torch
from tqdm.auto import tqdm

from ..constants import PYKEEN_MODULE
from ..typing import OneOrSequence
from ..utils import nested_get, rate_limited, upgrade_to_sequence
from ..version import get_version

__all__ = [
    "safe_diagonal",
    "adjacency_tensor_to_stacked_matrix",
    "use_horizontal_stacking",
    "ShapeError",
    # Caches
    "TextCache",
    "WikidataCache",
    "PyOBOCache",
]

logger = logging.getLogger(__name__)


def iter_matrix_power(matrix: torch.Tensor, max_iter: int) -> Iterable[torch.Tensor]:
    """
    Iterate over matrix powers.

    :param matrix: shape: `(n, n)`
        the square matrix
    :param max_iter:
        the maximum number of iterations.

    :yields: increasing matrix powers
    """
    yield matrix
    a = matrix
    for _ in range(max_iter - 1):
        # if the sparsity becomes too low, convert to a dense matrix
        # note: this heuristic is based on the memory consumption,
        # for a sparse matrix, we store 3 values per nnz (row index, column index, value)
        # performance-wise, it likely makes sense to switch even earlier
        # `torch.sparse.mm` can also deal with dense 2nd argument
        if a.is_sparse and a._nnz() >= a.numel() // 4:
            a = a.to_dense()
        # note: torch.sparse.mm only works for COO matrices;
        #       @ only works for CSR matrices
        if matrix.is_sparse_csr:
            a = matrix @ a
        else:
            a = torch.sparse.mm(matrix, a)
        yield a


def safe_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """
    Extract diagonal from a potentially sparse matrix.

    .. note ::
        this is a work-around as long as :func:`torch.diagonal` does not work for sparse tensors

    :param matrix: shape: `(n, n)`
        the matrix

    :return: shape: `(n,)`
        the diagonal values.
    """
    if not matrix.is_sparse:
        return torch.diagonal(matrix)

    # convert to COO, if necessary
    if matrix.is_sparse_csr:
        matrix = matrix.to_sparse_coo()

    n = matrix.shape[0]
    # we need to use indices here, since there may be zero diagonal entries
    indices = matrix._indices()
    mask = indices[0] == indices[1]
    diagonal_values = matrix._values()[mask]
    diagonal_indices = indices[0][mask]

    return torch.zeros(n, device=matrix.device).scatter_add(dim=0, index=diagonal_indices, src=diagonal_values)


def use_horizontal_stacking(
    input_dim: int,
    output_dim: int,
) -> bool:
    """
    Determine a stacking direction based on the input and output dimension.

    The vertical stacking approach is suitable for low dimensional input and high dimensional output,
    because the projection to low dimensions is done first. While the horizontal stacking approach is good
    for high dimensional input and low dimensional output as the projection to high dimension is done last.

    :param input_dim:
        the layer's input dimension
    :param output_dim:
        the layer's output dimension

    :return:
        whether to use horizontal (True) or vertical stacking

    .. seealso :: [thanapalasingam2021]_
    """
    return input_dim > output_dim


def adjacency_tensor_to_stacked_matrix(
    num_relations: int,
    num_entities: int,
    source: torch.LongTensor,
    target: torch.LongTensor,
    edge_type: torch.LongTensor,
    edge_weights: Optional[torch.FloatTensor] = None,
    horizontal: bool = True,
) -> torch.Tensor:
    """
    Stack adjacency matrices as described in [thanapalasingam2021]_.

    This method re-arranges the (sparse) adjacency tensor of shape
    `(num_entities, num_relations, num_entities)` to a sparse adjacency matrix of shape
    `(num_entities, num_relations * num_entities)` (horizontal stacking) or
    `(num_entities * num_relations, num_entities)` (vertical stacking). Thereby, we can perform the relation-specific
    message passing of R-GCN by a single sparse matrix multiplication (and some additional pre- and/or
    post-processing) of the inputs.

    :param num_relations:
        the number of relations
    :param num_entities:
        the number of entities
    :param source: shape: (num_triples,)
        the source entity indices
    :param target: shape: (num_triples,)
        the target entity indices
    :param edge_type: shape: (num_triples,)
        the edge type, i.e., relation ID
    :param edge_weights: shape: (num_triples,)
        scalar edge weights
    :param horizontal:
        whether to use horizontal or vertical stacking

    :return: shape: `(num_entities * num_relations, num_entities)` or `(num_entities, num_entities * num_relations)`
        the stacked adjacency matrix
    """
    offset = edge_type * num_entities
    if horizontal:
        size = (num_entities, num_relations * num_entities)
        target = offset + target
    else:
        size = (num_relations * num_entities, num_entities)
        source = offset + source
    indices = torch.stack([source, target], dim=0)
    if edge_weights is None:
        edge_weights = torch.ones_like(source, dtype=torch.get_default_dtype())
    return torch.sparse_coo_tensor(
        indices=indices,
        values=edge_weights,
        size=size,
    )


WIKIDATA_IMAGE_RELATIONS = [
    "P18",  # image
    "P948",  # page banner
    "P41",  # flag image
    "P94",  # coat of arms image
    "P154",  # logo image
    "P242",  # locator map image
]


class TextCache(ABC):
    """An interface for looking up text for various flavors of entity identifiers."""

    @abstractmethod
    def get_texts(self, identifiers: Sequence[str]) -> Sequence[Optional[str]]:
        """Get text for the given identifiers for the cache."""


class IdentityCache(TextCache):
    """
    A cache without functionality.

    Mostly used for testing.
    """

    # docstr-coverage: inherited
    def get_texts(self, identifiers: Sequence[str]) -> Sequence[Optional[str]]:  # noqa: D102
        return identifiers


class WikidataCache(TextCache):
    """A cache for requests against Wikidata's SPARQL endpoint."""

    #: Wikidata SPARQL endpoint. See https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service#Interfacing
    WIKIDATA_ENDPOINT = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"

    HEADERS: Dict[str, str] = {
        # cf. https://meta.wikimedia.org/wiki/User-Agent_policy
        "User-Agent": (
            f"PyKEEN-Bot/{get_version()} (https://pykeen.github.io; pykeen2019@gmail.com) "
            f"requests/{requests.__version__}"
        ),
        # cf. https://wikitech.wikimedia.org/wiki/Robot_policy
        "Accept-Encoding": "gzip",
    }

    def __init__(self) -> None:
        """Initialize the cache."""
        self.module = PYKEEN_MODULE.module("wikidata")

    @staticmethod
    def verify_ids(ids: Sequence[str]):
        """
        Raise error if invalid IDs are encountered.

        :param ids:
            the ids to verify

        :raises ValueError:
            if any invalid ID is encountered
        """
        pattern = re.compile(r"Q(\d+)")
        invalid_ids = [one_id for one_id in ids if not pattern.match(one_id)]
        if invalid_ids:
            raise ValueError(f"Invalid IDs encountered: {invalid_ids}")

    @classmethod
    def query(
        cls,
        sparql: Union[str, Callable[..., str]],
        wikidata_ids: Sequence[str],
        batch_size: int = 256,
    ) -> Iterable[Mapping[str, Any]]:
        """
        Batched SPARQL query execution for the given IDS.

        :param sparql:
            the SPARQL query with a placeholder `ids`
        :param wikidata_ids:
            the Wikidata IDs
        :param batch_size:
            the batch size, i.e., maximum number of IDs per query

        :return:
            an iterable over JSON results, where the keys correspond to query variables,
            and the values to the corresponding binding
        """
        if not wikidata_ids:
            return {}

        if len(wikidata_ids) > batch_size:
            # break into smaller requests
            return chain.from_iterable(
                cls.query(sparql=sparql, wikidata_ids=id_batch, batch_size=batch_size)
                for id_batch in more_itertools.chunked(wikidata_ids, batch_size)
            )

        if isinstance(sparql, str):
            sparql = sparql.format
        sparql = sparql(ids=" ".join(f"wd:{i}" for i in wikidata_ids))
        logger.debug("running query: %s", sparql)
        res = requests.get(
            cls.WIKIDATA_ENDPOINT,
            params={"query": sparql, "format": "json"},
            headers=cls.HEADERS,
        )
        res.raise_for_status()
        bindings = res.json()["results"]["bindings"]
        logger.debug(f"Retrieved {len(bindings)} bindings")
        return bindings

    @classmethod
    def query_text(
        cls, wikidata_ids: Sequence[str], language: str = "en", batch_size: int = 256
    ) -> Mapping[str, Mapping[str, str]]:
        """
        Query the SPARQL endpoints about information for the given IDs.

        :param wikidata_ids:
            the Wikidata IDs
        :param language:
            the label language
        :param batch_size:
            the batch size; if more ids are provided, break the big request into multiple smaller ones

        :return:
            a mapping from Wikidata Ids to dictionaries with the label and description of the entities
        """
        res_json = cls.query(
            sparql=functools.partial(
                dedent(
                    """
                        SELECT ?item ?itemLabel ?itemDescription WHERE {{{{
                            VALUES ?item {{ {ids} }}
                            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language}". }}
                        }}}}
                    """
                ).format,
                language=language,
            ),
            wikidata_ids=wikidata_ids,
            batch_size=batch_size,
        )
        result = {}
        for entry in res_json:
            wikidata_id = nested_get(entry, "item", "value", default="")
            assert isinstance(wikidata_id, str)  # for mypy
            wikidata_id = wikidata_id.rsplit("/", maxsplit=1)[-1]
            label = nested_get(entry, "itemLabel", "value", default="")
            assert isinstance(label, str)  # for mypy
            description = nested_get(entry, "itemDescription", "value", default="")
            assert isinstance(description, str)  # for mypy
            result[wikidata_id] = dict(label=label, description=description)
        return result

    def _load(self, wikidata_id: str, component: str) -> Optional[str]:
        """Load information about a Wikidata ID from JSON file."""
        name = f"{wikidata_id}.json"
        if not self.module.join(name=name).is_file():
            return None
        return self.module.load_json(name=name)[component]

    def _save(self, entries: Mapping[str, Mapping[str, str]]):
        """Save entries as JSON."""
        logger.info(f"Saving {len(entries)} entries to {self.module}")
        for wikidata_id, entry in entries.items():
            name = f"{wikidata_id}.json"
            self.module.dump_json(name=name, obj=entry)

    def _get(self, ids: Sequence[str], component: Literal["label", "description"]) -> Sequence[str]:
        """
        Get the requested component for the given IDs.

        .. note ::
            this method uses file-based caching to avoid excessive requests to the Wikidata API.

        :param ids:
            the Wikidata IDs
        :param component:
            the selected component

        :return:
            the selected component for each Wikidata ID
        """
        self.verify_ids(ids=ids)
        # try to load cached first
        result: List[Optional[str]] = [None] * len(ids)
        for i, wikidata_id in enumerate(ids):
            result[i] = self._load(wikidata_id=wikidata_id, component=component)
        # determine missing entries
        missing = [wikidata_id for wikidata_id, desc in zip(ids, result) if not desc]
        # retrieve information via SPARQL
        entries = self.query_text(wikidata_ids=missing)
        # save entries
        self._save(entries=entries)
        # fill missing descriptions
        w_to_i = {wikidata_id: i for i, wikidata_id in enumerate(ids)}
        for wikidata_id, entry in entries.items():
            result[w_to_i[wikidata_id]] = entry[component]
        # for mypy
        for item in result:
            assert isinstance(item, str)
        return cast(Sequence[str], result)

    def get_texts(self, identifiers: Sequence[str]) -> Sequence[str]:
        """Get a concatenation of the title and description for each Wikidata identifier.

        :param identifiers:
            the Wikidata identifiers, each starting with Q (e.g., ``['Q42']``)

        :return:
            the label and description for each Wikidata entity concatenated
        """
        # get labels & descriptions
        titles = self.get_labels(wikidata_identifiers=identifiers)
        descriptions = self.get_descriptions(wikidata_identifiers=identifiers)
        # compose labels
        return [f"{title}: {description}" for title, description in zip(titles, descriptions)]

    def get_labels(self, wikidata_identifiers: Sequence[str]) -> Sequence[str]:
        """
        Get entity labels for the given IDs.

        :param wikidata_identifiers:
            the Wikidata identifiers, each starting with Q (e.g., ``['Q42']``)

        :return:
            the label for each Wikidata entity
        """
        return self._get(ids=wikidata_identifiers, component="label")

    def get_descriptions(self, wikidata_identifiers: Sequence[str]) -> Sequence[str]:
        """
        Get entity descriptions for the given IDs.

        :param wikidata_identifiers:
            the Wikidata identifiers, each starting with Q (e.g., ``['Q42']``)

        :return:
            the description for each Wikidata entity
        """
        return self._get(ids=wikidata_identifiers, component="description")

    def _discover_images(self, extensions: Collection[str]) -> Mapping[str, pathlib.Path]:
        image_dir = self.module.join("images")
        return {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix in {f".{e}" for e in extensions}
        }

    def get_image_paths(
        self,
        ids: Sequence[str],
        extensions: Collection[str] = ("jpeg", "jpg", "gif", "png", "svg", "tif"),
        progress: bool = False,
    ) -> Sequence[Optional[pathlib.Path]]:
        """Get paths to images for the given IDs.

        :param ids:
            the Wikidata IDs.
        :param extensions:
            the allowed file extensions
        :param progress:
            whether to display a progress bar

        :return:
            the paths to images for the given IDs.
        """
        id_to_path = self._discover_images(extensions=extensions)
        missing = sorted(set(ids).difference(id_to_path.keys()))
        num_missing = len(missing)
        logger.info(
            f"Downloading images for {num_missing:,} entities. With the rate limit in place, "
            f"this will take at least {num_missing / 10:.2f} seconds.",
        )
        res_json = self.query(
            sparql=functools.partial(
                dedent(
                    """
                    SELECT ?item ?relation ?image
                    WHERE {{
                        VALUES ?item {{ {ids} }} .
                        ?item ?r ?image .
                        VALUES ?r {{ {relations} }}
                    }}
                """
                ).format,
                relations=" ".join(f"wdt:{r}" for r in WIKIDATA_IMAGE_RELATIONS),
            ),
            wikidata_ids=missing,
        )
        # we can have multiple images per entity -> collect image URLs per image
        images: Dict[str, Dict[str, List[str]]] = {}
        for entry in res_json:
            # entity ID
            wikidata_id = nested_get(entry, "item", "value", default="")
            assert isinstance(wikidata_id, str)  # for mypy
            wikidata_id = wikidata_id.rsplit("/", maxsplit=1)[-1]

            # relation ID
            relation_id = nested_get(entry, "relation", "value", default="")
            assert isinstance(relation_id, str)  # for mypy
            relation_id = relation_id.rsplit("/", maxsplit=1)[-1]

            # image URL
            image_url = nested_get(entry, "image", "value", default=None)
            assert image_url is not None
            images.setdefault(wikidata_id, dict()).setdefault(relation_id, []).append(image_url)

        # check whether images are still missing
        missing = sorted(set(missing).difference(images.keys()))
        if missing:
            logger.warning(f"Could not retrieve an image URL for {len(missing)} entities: {missing}")

        # select on image url per image in a reproducible way
        for wikidata_id, url_dict in tqdm(rate_limited(images.items(), min_avg_time=0.1), disable=not progress):
            # traverse relations in order of preference
            for relation in WIKIDATA_IMAGE_RELATIONS:
                if relation not in url_dict:
                    continue
                # now there is an image available -> select reproducible by URL sorting
                image_url = sorted(url_dict[relation])[0]
                ext = image_url.rsplit(".", maxsplit=1)[-1].lower()
                if ext not in extensions:
                    logger.warning(f"Unknown extension: {ext} for {image_url}")
                self.module.ensure(
                    "images",
                    url=image_url,
                    name=f"{wikidata_id}.{ext}",
                    download_kwargs=dict(backend="requests", headers=self.HEADERS),
                )
            else:
                # did not break -> no image
                logger.warning(f"No image for {wikidata_id}")

        id_to_path = self._discover_images(extensions=extensions)
        return [id_to_path.get(i) for i in ids]


PYOBO_PREFIXES_WARNED = set()


class PyOBOCache(TextCache):
    """A cache that looks up labels of biomedical entities based on their CURIEs."""

    def __init__(self, *args, **kwargs):
        """Instantiate the PyOBO cache, ensuring PyOBO is installed."""
        try:
            import pyobo
        except ImportError:
            raise ImportError(f"Can not use {self.__class__.__name__} because pyobo is not installed.")
        else:
            self._get_name = pyobo.get_name
        super().__init__(*args, **kwargs)

    def get_texts(self, identifiers: Sequence[str]) -> Sequence[Optional[str]]:
        """Get text for the given CURIEs.

        :param identifiers:
            The compact URIs for each entity (e.g., ``['doid:1234', ...]``)

        :return:
            the label for each entity, looked up via :func:`pyobo.get_name`.
            Might be none if no label is available.
        """
        # This import doesn't need a wrapper since it's a transitive
        # requirement of PyOBO
        import bioregistry

        res: List[Optional[str]] = []
        for curie in identifiers:
            try:
                prefix, identifier = curie.split(":", maxsplit=1)
            except ValueError:
                res.append(None)
                continue

            norm_prefix = bioregistry.normalize_prefix(prefix)
            if norm_prefix is None:
                if prefix not in PYOBO_PREFIXES_WARNED:
                    logger.warning("Prefix not registered in the Bioregistry: %s", prefix)
                    PYOBO_PREFIXES_WARNED.add(prefix)
                res.append(None)
                continue

            try:
                name = self._get_name(norm_prefix, identifier)
            except subprocess.CalledProcessError:
                if norm_prefix not in PYOBO_PREFIXES_WARNED:
                    logger.warning("could not get names from %s", norm_prefix)
                    PYOBO_PREFIXES_WARNED.add(norm_prefix)
                res.append(None)
                continue
            else:
                res.append(name)
        return res


class ShapeError(ValueError):
    """An error for a mismatch in shapes."""

    def __init__(self, shape: Sequence[int], reference: Sequence[int]) -> None:
        """
        Initialize the error.

        :param shape: the mismatching shape
        :param reference: the expected shape
        """
        super().__init__(f"shape {shape} does not match expected shape {reference}")

    @classmethod
    def verify(cls, shape: OneOrSequence[int], reference: Optional[OneOrSequence[int]]) -> Sequence[int]:
        """
        Raise an exception if the shape does not match the reference.

        This method normalizes the shapes first.

        :param shape:
            the shape to check
        :param reference:
            the reference shape. If None, the shape always matches.

        :raises ShapeError:
            if the two shapes do not match.

        :return:
            the normalized shape
        """
        shape = upgrade_to_sequence(shape)
        if reference is None:
            return shape
        reference = upgrade_to_sequence(reference)
        if reference != shape:
            # darglint does not like
            # raise cls(shape=shape, reference=reference)
            raise ShapeError(shape=shape, reference=reference)
        return shape
