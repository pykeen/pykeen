"""Using text representations."""

import torch

from pykeen.datasets import get_dataset
from pykeen.models import ERModel
from pykeen.nn import TextRepresentation
from pykeen.pipeline import pipeline

dataset = get_dataset(dataset="nations")
entity_representations = TextRepresentation.from_dataset(dataset=dataset, encoder="transformer")
result = pipeline(
    dataset=dataset,
    model=ERModel,
    model_kwargs=dict(
        interaction="ermlpe",
        interaction_kwargs=dict(
            embedding_dim=entity_representations.shape[0],
        ),
        entity_representations=entity_representations,
        relation_representations_kwargs=dict(
            shape=entity_representations.shape,
        ),
    ),
    training_kwargs=dict(num_epochs=1),
)
model = result.model

# representations for unseen
entity_representation = model.entity_representations[0]
label_encoder = entity_representation.encoder
uk, united_kingdom = label_encoder(labels=["uk", "united kingdom"])

# true triple from train: ['brazil', 'exports3', 'uk']
relation_representation = model.relation_representations[0]
h_repr = entity_representation.get_in_more_canonical_shape(
    dim="h", indices=torch.as_tensor(dataset.entity_to_id["brazil"]).view(1)
)
r_repr = relation_representation.get_in_more_canonical_shape(
    dim="r", indices=torch.as_tensor(dataset.relation_to_id["exports3"]).view(1)
)
scores = model.interaction(h=h_repr, r=r_repr, t=torch.stack([uk, united_kingdom]))
print(scores)
