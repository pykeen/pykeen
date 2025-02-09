"""Using training callbacks."""

from pykeen.datasets import get_dataset
from pykeen.pipeline import pipeline

dataset = get_dataset(dataset="nations")
result = pipeline(
    dataset=dataset,
    model="mure",
    training_kwargs=dict(
        num_epochs=100,
        callbacks="evaluation",
        callbacks_kwargs=dict(
            evaluation_triples=dataset.training.mapped_triples,
            tracker="console",
            prefix="training",
        ),
    ),
)
