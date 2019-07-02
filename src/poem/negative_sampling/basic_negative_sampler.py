# -*- coding: utf-8 -*-

"""Negative sampling algorithm based on the work og of Bordes *et al.*."""

import numpy as np

from .base import NegativeSampler
from ..utils import slice_triples

__all__ = [
    'BasicNegativeSampler',
]


class BasicNegativeSampler(NegativeSampler):
    def __init__(self, all_entities):
        self.all_entities = all_entities
        self.num_entities = self.all_entities.shape[0]

    def sample(self, positive_batch) -> np.ndarray:
        batch_heads, batch_relations, batch_tails = slice_triples(positive_batch)

        num_subj_corrupt = len(positive_batch) // 2
        num_obj_corrupt = len(positive_batch) - num_subj_corrupt
        # pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=self.model.device)

        corrupted_subj_indices = np.random.choice(np.arange(0, self.num_entities), size=num_subj_corrupt)
        corrupted_subjects = np.reshape(self.all_entities[corrupted_subj_indices], newshape=(-1, 1))
        subject_based_corrupted_triples = np.concatenate(
            [
                corrupted_subjects,
                batch_relations[:num_subj_corrupt],
                batch_tails[:num_subj_corrupt],
            ],
            axis=1,
        )

        corrupted_obj_indices = np.random.choice(np.arange(0, self.num_entities), size=num_obj_corrupt)
        corrupted_objects = np.reshape(self.all_entities[corrupted_obj_indices], newshape=(-1, 1))

        object_based_corrupted_triples = np.concatenate(
            [
                batch_heads[num_subj_corrupt:],
                batch_relations[num_subj_corrupt:],
                corrupted_objects],
            axis=1,
        )

        neg_batch = np.concatenate(
            [
                subject_based_corrupted_triples,
                object_based_corrupted_triples,
            ],
            axis=0,
        )

        return neg_batch
