import torch
from pykeen.pipeline import pipeline


num_entities = 14
pretrained_embedding_tensor = torch.rand(num_entities, 128)


def initialize_from_pretrained(x: torch.FloatTensor) -> torch.FloatTensor:
    return pretrained_embedding_tensor


result = pipeline(
    dataset="nations",
    model="transe",
    model_kwargs=dict(
        embedding_dim=pretrained_embedding_tensor.shape[-1],
        entity_initializer=initialize_from_pretrained,
    ),
)
