"""Use the (generalized) low-rank approximation to create a mixture model representation."""

# %%
from pykeen.datasets import get_dataset
from pykeen.models import ERModel
from pykeen.nn import LowRankRepresentation
from pykeen.pipeline import pipeline

dataset = get_dataset(dataset="CoDExSmall")

# set up relation representations as a mixture (~soft clustering) with 5 components
embedding_dim = 32
num_components = 5
r = LowRankRepresentation(
    max_id=dataset.num_relations,
    shape=embedding_dim,
    num_bases=num_components,
    weight_kwargs=dict(normalizer="softmax"),
)
# use DistMult interaction, and a simple embedding matrix for relations
model = ERModel(
    triples_factory=dataset.training,
    interaction="distmult",
    entity_representations_kwargs=dict(shape=embedding_dim),
    relation_representations=r,
)
result = pipeline(dataset=dataset, model=model, training_kwargs=dict(num_epochs=20))

# %%
# TODO: get labels for relations
# e.g. https://www.wikidata.org/wiki/Property:P3373
# WikidataTextCache

# %%
# use the mixture weights
weights = r.weight()
component_index = weights.argmax(dim=1).tolist()
components = [[] for _ in range(num_components)]
for label, relation_index in dataset.relation_to_id.items():
    components[component_index[relation_index]].append(label)
print(components)
