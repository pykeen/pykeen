"""Example for backfill representations."""

from pykeen.datasets import get_dataset
from pykeen.nn import BackfillRepresentation, Embedding, init
from pykeen.pipeline import pipeline

# Here we simulate going from a smaller dataset to a larger one
# by starting with a large dataset and narrowing it down.
dataset_large = get_dataset(dataset="nations")
dataset = dataset_large.restrict(entities=["burma", "china", "india", "indonesia"])
# You can take a look at the datasets via `dataset.summarize()``

# Now we train a model on the small dataset.
embedding_dim = 32
result = pipeline(
    dataset=dataset,
    interaction="distmult",
    dimensions=dict(d=embedding_dim),
)

# We want to re-use the entity representations for known entities, without further training them.
old_repr: Embedding = result.model.entity_representations[0]
initializer = init.PretrainedInitializer(tensor=old_repr())

# The known IDs are the entity IDs of known entities *in the new entity to ID mapping*.
# Here, we look them up via their (unique) label.
known_ids = dataset_large.training.entities_to_ids(dataset.entity_to_id)
base_repr = Embedding(max_id=len(known_ids), shape=(embedding_dim,), initializer=initializer, trainable=False)

# Next, we directly create representations for the new entities using the backfill representation.
# To do this, we need to create an iterable (e.g., a set) of all of the entity IDs that are in the base
# representation. Then, the assignments to the base representation and an auxillary representation are
# automatically generated for the base class.
entity_repr = BackfillRepresentation(base_ids=known_ids, max_id=dataset_large.num_entities, base=base_repr)

# You can see the composition by printing the module.
print(entity_repr)

# Note: For simplify, we train new relation representations from scratch.
# You could also re-use the relation representations (if the set of relations stayed the same),
# or apply the same backfill method from above.

# The combined representation can now be used as any other representation, e.g., to train a model with
# distmult interaction:
result = pipeline(
    dataset=dataset_large,
    interaction="distmult",
    dimensions=dict(d=embedding_dim),
    model_kwargs=dict(entity_representations=entity_repr),
)
