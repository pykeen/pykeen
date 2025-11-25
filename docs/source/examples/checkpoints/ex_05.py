"""Write a checkpoint every 10 steps, but keep only the last one and one every 50 steps."""

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
            "keeper": "union",
            "keeper_kwargs": {
                "bases": ["modulo", "last"],
                "bases_kwargs": [{"divisor": 50}, None],
            },
        },
    },
)
