"""Write a checkpoint every 10 steps and keep them all."""

from pykeen.pipeline import pipeline

result = pipeline(
    dataset="nations",
    model="mure",
    training_kwargs={
        "num_epochs": 100,
        "callbacks": "checkpoint",
        # create one checkpoint every 10 epochs
        "callbacks_kwargs": {
            "schedule": "every",
            "schedule_kwargs": {
                "frequency": 10,
            },
        },
    },
)
