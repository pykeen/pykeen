# -*- coding: utf-8 -*-

"""Utilities for neural network components."""

import logging
from typing import Iterable, Optional, Sequence, Union

import torch
from more_itertools import chunked
from torch import nn
from torch_max_mem import MemoryUtilizationMaximizer
from tqdm.auto import tqdm

from ..utils import get_preferred_device, resolve_device, upgrade_to_sequence

__all__ = [
    "TransformerEncoder",
    "safe_diagonal",
    "adjacency_tensor_to_stacked_matrix",
    "use_horizontal_stacking",
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
        Initialize the encoder.

        :param pretrained_model_name_or_path:
            the name of the pretrained model, or a path, cf. :func:`transformers.AutoModel.from_pretrained`
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
        this is a work-around as long as `torch.diagonal` does not work for sparse tensors

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
