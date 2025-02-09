"""Write a checkpoint at epoch 1, 7, and 10 and keep them all."""

from pykeen.pipeline import pipeline

result = pipeline(
    dataset="nations",
    model="mure",
    training_kwargs=dict(
        num_epochs=10,
        callbacks="checkpoint",
        # create checkpoints at epoch 1, 7, and 10
        callbacks_kwargs=dict(
            schedule="explicit",
            schedule_kwargs=dict(steps=(1, 7, 10)),
        ),
    ),
)
