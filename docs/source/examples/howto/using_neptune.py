"""Using Neptune.ai as a result tracker."""

# %%
from pykeen.pipeline import pipeline

pipeline_result = pipeline(
    model="RotatE",
    dataset="Kinships",
    result_tracker="neptune",
    result_tracker_kwargs={
        "project_qualified_name": "cthoyt/sandbox",
        "experiment_name": "Tutorial Training of RotatE on Kinships",
    },
)

# %%
experiment_id = 4  # if doesn't already exist, will throw an error!
pipeline_result = pipeline(
    model="RotatE",
    dataset="Kinships",
    result_tracker="neptune",
    result_tracker_kwargs={
        "project_qualified_name": "cthoyt/sandbox",
        "experiment_id": 4,
    },
)
