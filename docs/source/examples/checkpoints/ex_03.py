"""Write a checkpoint avery 5 epochs, but also at epoch 7."""

from pykeen.pipeline import pipeline

result = pipeline(
    dataset="nations",
    model="mure",
    training_kwargs=dict(
        num_epochs=10,
        callbacks="checkpoint",
        callbacks_kwargs=dict(
            schedule="union",
            # create checkpoints every 5 epochs, and at epoch 7
            schedule_kwargs=dict(bases=["every", "explicit"], bases_kwargs=[dict(frequency=5), dict(steps=[7])]),
        ),
    ),
)
