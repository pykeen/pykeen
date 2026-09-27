"""Training with the batch-local closed world assumption."""

from pykeen.pipeline import pipeline

result = pipeline(
    dataset="CodexSmall",
    model="DistMult",
    model_kwargs={"embedding_dim": 32, "create_inverse_triples": True},
    loss="BCEWithLogits",
    training_loop="BatchCWA",
    training_kwargs={"num_epochs": 10, "batch_size": 256},
)
