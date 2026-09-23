"""Using file-based result trackers."""

# %%
from pykeen.pipeline import pipeline

pipeline_result = pipeline(
    model="RotatE",
    dataset="Kinships",
    result_tracker="csv",
)

# %%
pipeline_result = pipeline(
    model="RotatE",
    dataset="Kinships",
    result_tracker="csv",
    result_tracker_kwargs={
        "name": "test.csv",
    },
)

# %%
pipeline_result = pipeline(
    model="RotatE",
    dataset="Kinships",
    result_tracker="json",
    result_tracker_kwargs={
        "name": "test.json",
    },
)
