"""Troubleshooting model loading across PyKEEN versions."""

# %%
# If the model class structure has changed, first save the model's ``state_dict`` using the version of PyKEEN used
# for training.
import torch

from pykeen.pipeline import pipeline

result = pipeline(dataset="Nations", model="RotatE")
torch.save(result.model.state_dict(), "v1.7.0/model.state_dict.pt")

# %%
# Then, using the version of PyKEEN you want to use, instantiate the model and load the state dict.
from pykeen.datasets import get_dataset
from pykeen.models import RotatE

dataset = get_dataset(dataset="Nations")
model = RotatE(triples_factory=dataset.training)
state_dict = torch.load("v1.7.0/model.state_dict.pt")
model.load_state_dict(state_dict)

# %%
# If the model weight names have changed, you need to inspect the state-dict dictionaries in the different
# versions, and try to match the keys. Then modify the state dict accordingly before loading it. For example:
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
