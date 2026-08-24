"""A toy example with translational distance models."""

# %%
import numpy as np

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

triples = np.array(
    [
        ["Brussels", "locatedIn", "Belgium"],
        ["Belgium", "partOf", "EU"],
        ["EU", "hasCapital", "Brussels"],
    ]
)
tf = TriplesFactory.from_labeled_triples(triples)
results = pipeline(
    training=tf,
    testing=tf,
    model="TransE",
    model_kwargs={"embedding_dim": 2},
    random_seed=1,
    device="cpu",
)
results.plot()

# %%
results = pipeline(
    training=tf,
    testing=tf,
    model="TransE",
    model_kwargs={"embedding_dim": 2},
    optimizer_kwargs={"lr": 1.0e-1},
    training_kwargs={"num_epochs": 128, "use_tqdm_batch": False},
    evaluation_kwargs={"use_tqdm": False},
    random_seed=1,
    device="cpu",
)
results.plot()

# %%
toy_results = pipeline(
    training=tf,
    testing=tf,
    model="TransE",
    loss="softplus",
    model_kwargs={"embedding_dim": 2},
    optimizer_kwargs={"lr": 1.0e-1},
    training_kwargs={"num_epochs": 128, "use_tqdm_batch": False},
    evaluation_kwargs={"use_tqdm": False},
    random_seed=1,
    device="cpu",
)
toy_results.plot()
