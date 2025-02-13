"""Write a checkpoint whenever a metric improves (here, just the training loss)."""

from pykeen.checkpoints import MetricSelection
from pykeen.pipeline import pipeline
from pykeen.trackers import tracker_resolver

# create a default result tracker (or use a proper one)
result_tracker = tracker_resolver.make(None)
result = pipeline(
    dataset="nations",
    model="mure",
    training_kwargs=dict(
        num_epochs=10,
        callbacks="checkpoint",
        callbacks_kwargs=dict(
            schedule="best",
            schedule_kwargs=dict(
                result_tracker=result_tracker,
                # in this example, we just use the training loss
                metric_selection=MetricSelection(
                    metric="loss",
                    maximize=False,
                ),
            ),
        ),
    ),
    # Important: use the same result tracker instance as in the checkpoint callback
    result_tracker=result_tracker,
)
