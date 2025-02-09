"""The partition representation."""

import torch

from pykeen.nn import Embedding, PartitionRepresentation, init
from pykeen.pipeline import pipeline
from pykeen.triples.generation import generate_triples_factory

num_entities = 5

# create embedding from label encodings
labels = {1: "a first description", 4: "a second description"}
label_initializer = init.LabelBasedInitializer(labels=list(labels.values()))
label_repr = label_initializer.as_embedding()
shape = label_repr.shape

# create a simple embedding matrix for all remaining ones
non_label_repr = Embedding(max_id=num_entities - len(labels), shape=shape)

# compose partition representation with a hard-coded assignment
assignment = torch.as_tensor([(1, 0), (0, 0), (1, 1), (1, 2), (0, 1)])
entity_repr = PartitionRepresentation(assignment=assignment, bases=[label_repr, non_label_repr])

# For brevity, we use here randomly generated triples factories instead of the actual data
training = generate_triples_factory(num_entities=num_entities, num_relations=5, num_triples=31)
testing = generate_triples_factory(num_entities=num_entities, num_relations=5, num_triples=17)

# we can use this to train a model
pipeline(
    interaction="distmult",
    dimensions={"d": shape[0]},
    model_kwargs=dict(entity_representations=entity_repr),
    training=training,
    testing=testing,
)
