"""Training with relation-specific loss weights."""

from pykeen.datasets.utils import get_dataset
from pykeen.pipeline import pipeline
from pykeen.triples.weights import RelationLossWeighter

dataset = get_dataset(dataset="CodexSmall")
loss_weighter = RelationLossWeighter.inverse_relation_frequency(mapped_triples=dataset.training.mapped_triples)
result = pipeline(
    dataset=dataset,
    model="MuRE",
    loss="BCEWithLogits",
    training_loop_kwargs=dict(loss_weighter=loss_weighter),
    training_loop="lcwa",
)
