"""Training a Model with the Pipeline."""

# %%
from pykeen.pipeline import pipeline

pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
)
pipeline_result.save_to_directory("nations_transe")

# %%
from pykeen.models import TransE

pipeline_result = pipeline(
    dataset="Nations",
    model=TransE,
)
pipeline_result.save_to_directory("nations_transe")

# %%
from pykeen.datasets import Nations

pipeline_result = pipeline(
    dataset=Nations,
    model=TransE,
)
pipeline_result.save_to_directory("nations_transe")

# %%
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    training_loop="sLCWA",
)
pipeline_result.save_to_directory("nations_transe")

# %%
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    training_loop="LCWA",
)
pipeline_result.save_to_directory("nations_transe")

# %%
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    training_loop="sLCWA",
    negative_sampler="basic",
)
pipeline_result.save_to_directory("nations_transe")

# %%
from pykeen.sampling import BasicNegativeSampler

pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    training_loop="sLCWA",
    negative_sampler=BasicNegativeSampler,
)
pipeline_result.save_to_directory("nations_transe")

# %%
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    evaluator="RankBasedEvaluator",
)
pipeline_result.save_to_directory("nations_transe")

# %%
from pykeen.evaluation import RankBasedEvaluator

pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    evaluator=RankBasedEvaluator,
)
pipeline_result.save_to_directory("nations_transe")

# %%
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    stopper="early",
)
pipeline_result.save_to_directory("nations_transe")

# %%
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    lr_scheduler="ExponentialLR",
    lr_scheduler_kwargs={
        "gamma": 0.99,
    },
)
pipeline_result.save_to_directory("nations_transe")

# %%
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    model_kwargs={
        "scoring_fct_norm": 2,
    },
)
pipeline_result.save_to_directory("nations_transe")
