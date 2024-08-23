from pykeen.pipeline import pipeline

result = pipeline(
    dataset="nations",
    model="mure",
    training_kwargs=dict(
        num_epochs=100,
        callbacks="checkpoint",
        # create one checkpoint every 10 epochs
        callbacks_kwargs=dict(
            schedule="every",
            schedule_kwargs=dict(
                frequency=10,
            ),
            keeper="union",
            keeper_kwargs=dict(
                bases=["modulo", "last"],
                bases_kwargs=[dict(divisor=50), None],
            )
        )
    ),
)