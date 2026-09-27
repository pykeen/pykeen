"""Using MLflow as a result tracker."""

# %%
from pykeen.pipeline import pipeline

pipeline_result = pipeline(
    model="RotatE",
    dataset="Kinships",
    result_tracker="mlflow",
    result_tracker_kwargs={
        "tracking_uri": "http://localhost:5000",
        "experiment_name": "Tutorial Training of RotatE on Kinships",
    },
)

# %%
from pykeen.hpo import hpo_pipeline

hpo_pipeline_result = hpo_pipeline(
    model="RotatE",
    dataset="Kinships",
    result_tracker="mlflow",
    result_tracker_kwargs={
        "tracking_uri": "http://localhost:5000",
        "experiment_name": "Tutorial HPO Training of RotatE on Kinships",
    },
)

# %%
experiment_id = 4  # if doesn't already exist, will throw an error!
pipeline_result = pipeline(
    model="RotatE",
    dataset="Kinships",
    result_tracker="mlflow",
    result_tracker_kwargs={
        "tracking_uri": "http://localhost:5000",
        "experiment_id": 4,
    },
)
