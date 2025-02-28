"""Training with batch-local closed world assumption."""

# %%
from pykeen.datasets.utils import get_dataset
from pykeen.models import ConvE
from pykeen.training.cwa import BatchCWATrainingLoop

# %%
dataset = get_dataset(dataset="CodexSmall")
model = ConvE(embedding_dim=256, triples_factory=dataset.training, loss="BCEWithLogits").to(device="cuda")
# %%
loop = BatchCWATrainingLoop(
    model=model,
    triples_factory=dataset.training,
    result_tracker="console",
    result_tracker_kwargs=dict(metric_filter=r".*both\.realistic\.hits_at_.*"),
    optimizer_kwargs=dict(lr=0.001),
)
# %%
loop.train(
    triples_factory=dataset.training,
    num_epochs=400,
    batch_size=64,
    callbacks="evaluation",
    callbacks_kwargs=dict(
        evaluation_triples=dataset.validation.mapped_triples,
        prefix="validation",
        additional_filter_triples=[dataset.training.mapped_triples],
    ),
)

# %%
