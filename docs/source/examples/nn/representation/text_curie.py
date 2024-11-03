"""Example for using biomedical CURIEs with text representations.."""

import bioontologies
import numpy as np

from pykeen.datasets.base import Dataset
from pykeen.models import ERModel
from pykeen.nn import BiomedicalCURIERepresentation
from pykeen.pipeline import pipeline
from pykeen.triples.triples_factory import TriplesFactory

# Generate graph dataset from the Monarch Disease Ontology (MONDO)
graph = bioontologies.get_obograph_by_prefix("mondo").squeeze(standardize=True)
triples = (edge.as_tuple() for edge in graph.edges)
triples = [t for t in triples if all(t)]
triples = TriplesFactory.from_labeled_triples(np.array(triples))
dataset = Dataset.from_tf(triples)

entity_representations = BiomedicalCURIERepresentation.from_dataset(
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
