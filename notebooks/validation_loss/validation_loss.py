# %%
"""Validation loss notebook."""

import pandas
import seaborn

from pykeen.datasets import get_dataset
from pykeen.pipeline import pipeline
from pykeen.trackers import PythonResultTracker

# %% [markdown]
# ## Training a model with PyKEEN

# %%
dataset = get_dataset(dataset="nations")
result_tracker = PythonResultTracker()
result = pipeline(
    dataset=dataset,
    model="mure",
    model_kwargs={"embedding_dim": 16},
    training_kwargs={
        "num_epochs": 100,
        # this will log a metric with name "validation.loss" to the configured result tracker
        "callbacks": "evaluation-loss",
        "callback_kwargs": {"triples_factory": dataset.validation, "prefix": "validation"},
    },
    result_tracker=result_tracker,
)
# %% [markdown]
# ## Evaluation with seaborn

# %%
grid = seaborn.relplot(
    data=pandas.DataFrame(
        data=[
            [step, step_metrics.get("loss"), step_metrics.get("validation.loss")]
            for step, step_metrics in result_tracker.metrics.items()
        ],
        columns=["step", "training", "validation"],
    )
    .set_index("step")
    .rolling(window=5)
    .agg(["min", "mean", "max"]),
    kind="line",
)
grid.fig.show()
