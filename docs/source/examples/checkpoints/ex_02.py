"""Write a checkpoint at epoch 1, 7, and 10 and keep them all."""

from pykeen.pipeline import pipeline

result = pipeline(
    dataset="nations",
    model="mure",
    training_kwargs={
        "num_epochs": 10,
        "callbacks": "checkpoint",
        # create checkpoints at epoch 1, 7, and 10
        "callbacks_kwargs": {
            "schedule": "explicit",
            "schedule_kwargs": {"steps": (1, 7, 10)},
        },
    },
)
