"""Example for backfill representations."""

import torch

from pykeen.datasets import get_dataset
from pykeen.nn import BackfillRepresentation, Embedding, init
from pykeen.pipeline import pipeline

dataset = get_dataset(dataset="nations")

# we start by creating the representation for those entities where we have pre-trained features
# here we simulate this for a set of Asian countries
embedding_dim = 32
known_ids = dataset.training.entities_to_ids(["burma", "china", "india", "indonesia"])
pre_trained_embeddings = torch.rand(len(known_ids), embedding_dim)
initializer = init.PretrainedInitializer(tensor=pre_trained_embeddings)
base_repr = Embedding(max_id=len(known_ids), shape=(embedding_dim,), initializer=initializer, trainable=False)

# Next, we directly create representations for the remaining ones using the backfill representation.
# To do this, we need to create an iterable (e.g., a set) of all of the entity IDs that are in the base
# representation. Then, the assignments to the base representation and an auxillary representation are
# automatically generated for the base class.
entity_repr = BackfillRepresentation(base_ids=known_ids, max_id=dataset.num_entities, base=base_repr)

# We assume that we do not have any pre-trained information for relations here for simplicity and train
# them from scratch.
relation_repr = Embedding(max_id=dataset.num_relations, shape=(embedding_dim,))

# The combined representation can now be used as any other representation, e.g., to train a model with
# distmult interaction:
result = pipeline(
    dataset=dataset,
    interaction="distmult",
    dimensions=dict(d=embedding_dim),
    model_kwargs=dict(
        entity_representations=entity_repr,
        relation_representations=relation_repr,
    ),
)
