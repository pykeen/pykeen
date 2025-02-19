"""Use the (generalized) low-rank approximation to create a mixture model representation."""

import pandas

from pykeen.datasets import get_dataset
from pykeen.models import ERModel
from pykeen.nn import LowRankRepresentation
from pykeen.nn.text.cache import WikidataTextCache
from pykeen.pipeline import pipeline
from pykeen.typing import FloatTensor

dataset = get_dataset(dataset="CoDExSmall", dataset_kwargs=dict(create_inverse_triples=True))

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

# keys are Wikidata IDs, which are the "labels" in CoDEx, and values
# are the concatenation of the Wikidata label + description
relation_to_text = WikidataTextCache().get_texts_dict(dataset.relation_to_id)

# use the mixture weights
weights = r.weight().detach().cpu().numpy()
_data = []
for label, relation_index in dataset.relation_to_id.items():
    text = relation_to_text[label]
    for c, w in enumerate(weights[relation_index]):
        _data.append((relation_index, label, text, c, w))
df = pandas.DataFrame(data=_data, columns=["relation_index", "wikidata-id", "text", "component_index", "weight"])


# For each component, look at the relations that are most assigned to it
print(
    df.groupby(by="component_index").apply(lambda g: g.nlargest(3, columns="weight"))[["wikidata-id", "text", "weight"]]
)
