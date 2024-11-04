"""Monte-Carlo uncertainty estimation with embedding dropout."""

import torch

from pykeen.datasets import Nations
from pykeen.models import ERModel
from pykeen.typing import FloatTensor

dataset = Nations()
model: ERModel[FloatTensor, FloatTensor, FloatTensor] = ERModel(
    triples_factory=dataset.training,
    interaction="distmult",
    entity_representations_kwargs=dict(embedding_dim=3, dropout=0.1),
    relation_representations_kwargs=dict(embedding_dim=3, dropout=0.1),
)
batch = torch.as_tensor(data=[[0, 1, 0]]).repeat(10, 1)
scores = model.score_hrt(batch)
