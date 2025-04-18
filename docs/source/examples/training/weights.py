"""Training with sample weights."""

from pykeen.datasets.utils import get_dataset
from pykeen.pipeline import pipeline
from pykeen.triples.weights import RelationSampleWeighter

dataset = get_dataset(dataset="CodexSmall")
sample_weighter = RelationSampleWeighter.inverse_relation_frequency(mapped_triples=dataset.training.mapped_triples)
result = pipeline(
    dataset=dataset,
    model="MuRE",
    loss="BCEWithLogits",
    training_loop_kwargs=dict(sample_weighter=sample_weighter),
    training_loop="lcwa",
)
