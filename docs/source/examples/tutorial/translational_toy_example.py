"""A toy example with translational distance models."""

# %%
# Given the following toy example comprising three entities in a triangle, a translational distance model like
# :class:`~pykeen.models.TransE` should be able to exactly learn the geometric structure.
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
# First, check if the model is converging using ``results.plot_losses``. Qualitatively, this means that the loss is
# smoothly decreasing and eventually evening out. If the model does not decrease, you might need to tune some
# parameters with the ``optimizer_kwargs`` and ``training_kwargs`` to the ``pipeline()`` function.
#
# For example, you can decrease the optimizer's learning rate to make the loss curve less bumpy. Second, you can
# increase the number of epochs during training.
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
# Please notice that there is some stochasticity in the training, since we sample negative examples for positive
# ones. Thus, the loss may fluctuate naturally. To better see the trend, you can smooth the loss by averaging over
# a window of epochs.
#
# We use a margin-based loss with TransE by default. Thus, it suffices if the model predicts scores such that the
# scores of positive triples and negative triples are at least one margin apart. Once the model has reached this
# state, it will not improve further upon these examples, as the embeddings are "good enough". Hence, an optimal
# solution with margin-based loss might not look like the exact geometric solution. If you want to change that you
# can switch to a loss function which does not use a margin, e.g. the softplus loss. You can do this by passing
# ``loss="softplus"`` to the pipeline.
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
