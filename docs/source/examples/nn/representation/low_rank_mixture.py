"""Use the (generalized) low-rank approximation to create a mixture model representation."""

# %%
from pykeen.datasets import get_dataset
from pykeen.models import ERModel
from pykeen.nn import LowRankRepresentation
from pykeen.nn.text.cache import WikidataTextCache
from pykeen.pipeline import pipeline
from pykeen.typing import FloatTensor

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
model = ERModel[FloatTensor, FloatTensor, FloatTensor](
    triples_factory=dataset.training,
    interaction="distmult",
    entity_representations_kwargs=dict(shape=embedding_dim),
    relation_representations=r,
)
result = pipeline(dataset=dataset, model=model, training_kwargs=dict(num_epochs=20))

# %%
# TODO: use this relation to label/description dict
# e.g. https://www.wikidata.org/wiki/Property:P3373

# keys are Wikidata IDs, which are the "labels" in CoDEx, and values
# are the concatenation of the Wikidata label + description
relation_to_text: dict[str, str | None] = WikidataTextCache().get_texts_dict(dataset.relation_to_id)

# %%
# use the mixture weights
weights = r.weight()
component_index = weights.argmax(dim=1).tolist()
components: list[list[str]] = [[] for _ in range(num_components)]
for label, relation_index in dataset.relation_to_id.items():
    components[component_index[relation_index]].append(label)
print(components)
