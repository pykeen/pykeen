"""Training with batch-local closed world assumption."""

# %%
import torch

from pykeen.datasets.utils import get_dataset
from pykeen.models import DistMult
from pykeen.training.bcwa import BatchCWATrainingLoop

# %%
dataset = get_dataset(dataset="CodexSmall", dataset_kwargs=dict(create_inverse_triples=True))
model = DistMult(embedding_dim=32, triples_factory=dataset.training, loss="BCEWithLogits").to(device="cuda")
# %%
loop = BatchCWATrainingLoop(
    model=model,
    triples_factory=dataset.training,
    result_tracker="console",
    result_tracker_kwargs=dict(metric_filter=r".*both\.realistic\.adjusted.*"),
)
# %%
validation = dataset.validation
training_triples = dataset.training.mapped_triples
partial_training_triples = training_triples[torch.randperm(dataset.training.num_triples)[:1000]]
assert validation is not None
loop.train(
    triples_factory=dataset.training,
    num_epochs=10,
    batch_size=256,
    callbacks=["evaluation"] * 2,
    callbacks_kwargs=[
        dict(
            evaluation_triples=validation.mapped_triples,
            prefix="validation",
            additional_filter_triples=[dataset.training.mapped_triples],
        ),
        dict(
            evaluation_triples=partial_training_triples,
            prefix="training",
            additional_filter_triples=[dataset.training.mapped_triples],
        ),
    ],
)

# TODO: Loss seems to decrease, but metrics *decrease* (to negative adjusted indices)
#  That looks like something is wrong somewhere

# %%
