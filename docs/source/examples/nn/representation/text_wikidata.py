"""Example for using WikidataTextRepresentation."""

from pykeen.datasets import get_dataset
from pykeen.models import ERModel
from pykeen.nn import WikidataTextRepresentation
from pykeen.pipeline import pipeline

dataset = get_dataset(dataset="codexsmall")
entity_representations = WikidataTextRepresentation.from_dataset(
    dataset=dataset,
    encoder="transformer",
)

result = pipeline(
    dataset=dataset,
    model=ERModel,
    model_kwargs=dict(
        interaction="distmult",
        entity_representations=entity_representations,
        relation_representation_kwargs=dict(
            shape=entity_representations.shape,
        ),
    ),
)
