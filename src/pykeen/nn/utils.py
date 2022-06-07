# -*- coding: utf-8 -*-

"""Utilities for neural network components."""

from __future__ import annotations

import logging
import re
from itertools import chain
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Union, cast

import more_itertools
import requests
import torch
from more_itertools import chunked
from torch import nn
from torch_max_mem import MemoryUtilizationMaximizer
from tqdm.auto import tqdm

from ..constants import PYKEEN_MODULE
from ..utils import get_preferred_device, resolve_device, upgrade_to_sequence
from ..version import get_version

__all__ = [
    "TransformerEncoder",
    "safe_diagonal",
    "adjacency_tensor_to_stacked_matrix",
    "use_horizontal_stacking",
    "WikidataCache",
]

logger = logging.getLogger(__name__)
memory_utilization_maximizer = MemoryUtilizationMaximizer()


@memory_utilization_maximizer
def _encode_all_memory_utilization_optimized(
    encoder: "TransformerEncoder",
    labels: Sequence[str],
    batch_size: int,
) -> torch.Tensor:
    """
    Encode all labels with the given batch-size.

    Wrapped by memory utilization maximizer to automatically reduce the batch size if needed.

    :param encoder:
        the encoder
    :param labels:
        the labels to encode
    :param batch_size:
        the batch size to use. Will automatically be reduced if necessary.

    :return: shape: `(len(labels), dim)`
        the encoded labels
    """
    return torch.cat(
        [encoder(batch) for batch in chunked(tqdm(map(str, labels), leave=False), batch_size)],
        dim=0,
    )


class TransformerEncoder(nn.Module):
    """A combination of a tokenizer and a model."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_length: Optional[int] = None,
    ):
        """
        Initialize the encoder using :class:`transformers.AutoModel`.

        :param pretrained_model_name_or_path:
            the name of the pretrained model, or a path, cf. :meth:`transformers.AutoModel.from_pretrained`
        :param max_length: >0, default: 512
            the maximum number of tokens to pad/trim the labels to

        :raises ImportError:
            if the :mod:`transformers` library could not be imported
        """
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer
        except ImportError as error:
            raise ImportError(
                "Please install the `transformers` library, use the _transformers_ extra"
                " for PyKEEN with `pip install pykeen[transformers] when installing, or "
                " see the PyKEEN installation docs at https://pykeen.readthedocs.io/en/stable/installation.html"
                " for more information."
            ) from error

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path).to(
            resolve_device()
        )
        self.max_length = max_length or 512

    def forward(self, labels: Union[str, Sequence[str]]) -> torch.FloatTensor:
        """Encode labels via the provided model and tokenizer."""
        labels = upgrade_to_sequence(labels)
        labels = list(map(str, labels))
        return self.model(
            **self.tokenizer(
                labels,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(get_preferred_device(self.model))
        ).pooler_output

    @torch.inference_mode()
    def encode_all(
        self,
        labels: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        """Encode all labels (inference mode & batched).

        :param labels:
            a sequence of strings to encode
        :param batch_size:
            the batch size to use for encoding the labels. ``batch_size=1``
            means that the labels are encoded one-by-one, while ``batch_size=len(labels)``
            would correspond to encoding all at once.
            Larger batch sizes increase memory requirements, but may be computationally
            more efficient. `batch_size` can also be set to `None` to enable automatic batch
            size maximization for the employed hardware.

        :returns: shape: (len(labels), dim)
            a tensor representing the encodings for all labels
        """
        return _encode_all_memory_utilization_optimized(
            encoder=self, labels=labels, batch_size=batch_size or len(labels)
        ).detach()


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


class WikidataCache:
    """A cache for requests against Wikidata's SPARQL endpoint."""

    #: Wikidata SPARQL endpoint. See https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service#Interfacing
    WIKIDATA_ENDPOINT = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"

    # image:
    # https://www.wikidata.org/w/api.php?action=wbgetclaims&property=P18&entity=Qxxx

    def __init__(self) -> None:
        """Initialize the cache."""
        self.module = PYKEEN_MODULE.submodule("wikidata")

    @staticmethod
    def iter_invalid_ids(ids: Sequence[str]) -> Iterable[str]:
        """Iterate over invalid IDs."""
        pattern = re.compile(r"Q(\d+)")
        for one_id in ids:
            if not pattern.match(one_id):
                yield one_id

    @staticmethod
    def _safe_get(d: dict, *keys: str, default=None) -> Optional[Any]:
        """
        Get from a nested dictionary with default.

        :param d:
            the (nested) dictionary
        :param keys:
            the sequence of keys
        :param default:
            the default value if a key is not present at any level, defaults to None
        :return: the value, or the default
        """
        # TODO: move to utils
        for key in keys[:-1]:
            d = d.get(key, {})
        return d.get(keys[-1], default)

    @classmethod
    def query(
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
        # cf. https://github.com/biopragmatics/bioregistry/blob/55584709e287d1d01d51375e0bd836f3c4d25b2e/src/bioregistry/utils.py#L53-L63  # noqa: E501
        if not wikidata_ids:
            return {}

        if len(wikidata_ids) > batch_size:
            # break into smaller requests
            return dict(
                chain.from_iterable(
                    cls.query(wikidata_ids=id_batch, language=language, batch_size=batch_size).items()
                    for id_batch in more_itertools.chunked(wikidata_ids, batch_size)
                )
            )

        qualified = " ".join(f"wd:{i}" for i in wikidata_ids)
        sparql = f"""
            SELECT ?item ?itemLabel ?itemDescription WHERE {{{{
                VALUES ?item {{ {qualified} }}
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language}". }}
            }}}}
        """

        logger.debug("running query: %s", sparql)
        res = requests.get(
            cls.WIKIDATA_ENDPOINT,
            params={"query": sparql, "format": "json"},
            headers={"User-Agent": f"pykeen/{get_version()} (https://pykeen.github.io)"},
        )
        res.raise_for_status()
        res_json = res.json()
        result = {}
        for entry in res_json["results"]["bindings"]:
            wikidata_id = cls._safe_get(entry, "item", "value", default="")
            assert isinstance(wikidata_id, str)  # for mypy
            wikidata_id = wikidata_id.rsplit("/", maxsplit=1)[-1]
            label = cls._safe_get(entry, "itemLabel", "value", default="")
            assert isinstance(label, str)  # for mypy
            description = cls._safe_get(entry, "itemDescription", "value", default="")
            assert isinstance(description, str)  # for mypy
            result[wikidata_id] = dict(label=label, description=description)
        return result

    def verify_ids(self, ids: Sequence[str]):
        """Raise error if invalid IDs are encountered."""
        invalid_ids = list(self.iter_invalid_ids(ids=ids))
        if invalid_ids:
            raise ValueError(f"Invalid IDs encountered: {invalid_ids}")

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

    def _get(self, ids: Sequence[str], component: str) -> Sequence[str]:
        """Get the requested component for the given IDs."""
        self.verify_ids(ids=ids)
        # try to load cached first
        result: List[Optional[str]] = [None] * len(ids)
        for i, wikidata_id in enumerate(ids):
            result[i] = self._load(wikidata_id=wikidata_id, component=component)
        # determine missing entries
        missing = [wikidata_id for wikidata_id, desc in zip(ids, result) if not desc]
        # retrieve information via SPARQL
        entries = self.query(wikidata_ids=missing)
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

    def get_labels(self, ids: Sequence[str]) -> Sequence[str]:
        """Get entity labels for the given IDs."""
        return self._get(ids=ids, component="label")

    def get_descriptions(self, ids: Sequence[str]) -> Sequence[str]:
        """Get entity descriptions for the given IDs."""
        return self._get(ids=ids, component="description")
