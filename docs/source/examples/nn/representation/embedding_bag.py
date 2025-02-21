"""Embedding bag, with 2048-dimensional Morgan boolean fingerprints for molecules."""

import chembl_downloader
import torch

from pykeen.nn.representation import EmbeddingBagRepresentation

chembl_ids, tensors = zip(
    *((chembl_id, torch.tensor(arr, dtype=torch.bool)) for chembl_id, arr in chembl_downloader.iterate_fps()),
    strict=False,
)

representation = EmbeddingBagRepresentation.from_iter(
    list(fingerprint.nonzero())  # might need to flatten here
    for fingerprint in tensors
)
