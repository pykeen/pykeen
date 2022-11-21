"""Base hyper-relational dataset classes."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..base import Dataset, UnpackedRemoteDataset
from ...triples import StatementFactory

__all__ = [
    "HyperRelationalDataset",
    "HyperRelationalUnpackedRemoteDataset",
]


class HyperRelationalDataset(Dataset):
    """A hyper-relational dataset."""

    training: StatementFactory
    testing: StatementFactory
    validation: Optional[StatementFactory]
    triples_factory_cls = StatementFactory


class HyperRelationalUnpackedRemoteDataset(UnpackedRemoteDataset, HyperRelationalDataset):
    """A remote, unpacked, hyper-relational dataset."""

    def __init__(
        self, *, max_num_qualifier_pairs: int = -1, load_triples_kwargs: Optional[Mapping[str, Any]] = None, **kwargs
    ):
        """Initialize dataset.

        :param load_triples_kwargs:
            Arguments to pass through to :func:`StatementFactory.from_path`
            and ultimately through to :func:`pykeen.triples.utils.load_triples`.
        :param max_num_qualifier_pairs:
            TODO migalkin
        :param kwargs:
            Keyword arguments to pass to parent constructor
        """
        self.max_num_qualifier_pairs = max_num_qualifier_pairs

        # TODO the only difference with vanilla UnpackedRemoteDataset is here:
        # we update the kwargs with the max number of qualifier pairs to keep
        load_triples_kwargs = dict(load_triples_kwargs or {})
        load_triples_kwargs["max_num_qualifier_pairs"] = max_num_qualifier_pairs

        super().__init__(load_triples_kwargs=load_triples_kwargs, **kwargs)
