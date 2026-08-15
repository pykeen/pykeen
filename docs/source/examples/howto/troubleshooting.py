"""Troubleshooting model loading across PyKEEN versions."""

# %%
import torch

from pykeen.pipeline import pipeline

result = pipeline(dataset="Nations", model="RotatE")
torch.save(result.model.state_dict(), "v1.7.0/model.state_dict.pt")

# %%
from pykeen.datasets import get_dataset
from pykeen.models import RotatE

dataset = get_dataset(dataset="Nations")
model = RotatE(triples_factory=dataset.training)
state_dict = torch.load("v1.7.0/model.state_dict.pt")
model.load_state_dict(state_dict)

# %%
state_dict = torch.load("v1.7.0/model.state_dict.pt")
# these are some example changes in weight names for RotatE between two different pykeen versions
for old_name, new_name in [
    (
        "entity_embeddings._embeddings.weight",
        "entity_representations.0._embeddings.weight",
    ),
    (
        "relation_embeddings._embeddings.weight",
        "relation_representations.0._embeddings.weight",
    ),
]:
    state_dict[new_name] = state_dict.pop(old_name)
# in this example, the new model does not have a regularizer, so we need to delete corresponding data
for name in ["regularizer.weight", "regularizer.regularization_term"]:
    state_dict.pop(name)
model.load_state_dict(state_dict)
