"""Mapping Entity and Relation Identifiers to their Names."""

# %%
import torch

from pykeen.datasets.utils import get_dataset
from pykeen.triples.triples_factory import TriplesFactory

# As an example, we will use a small dataset that comes with entity and relation labels.
dataset = get_dataset(dataset="nations")
triples_factory = dataset.training

# This dataset provides entity names.
assert isinstance(triples_factory, TriplesFactory)

# %%
# Direct access to the mapping, here for entities.
entity_labeling = triples_factory.entity_labeling
# a mapping from labels/strings to the Ids
print(entity_labeling.label_to_id)
# the inverse mapping
print(entity_labeling.id_to_label)

# %%
# The labeling object also offers convenience methods for converting ids in different formats to strings
entity_labeling.label(ids=1)
entity_labeling.label(ids=[1, 3])
entity_labeling.label(ids=torch.as_tensor([7]))


# %%
# The triples factory exposes utility methods to normalize to ids
ids = triples_factory.entities_to_ids(entities=[3, 2])
ids = triples_factory.entities_to_ids(entities=["cuba", "china"])
# TODO: we should move that to the labeling

# %%
# Get tensor of entity identifiers
entity_ids = torch.as_tensor(triples_factory.entities_to_ids(["china", "egypt"]))
relation_ids = torch.as_tensor(triples_factory.relations_to_ids(["independence", "embassy"]))
