"""Using transformed representations."""

from torch import nn

from pykeen.datasets import get_dataset
from pykeen.nn import TransformedRepresentation, init

dataset = get_dataset(dataset="nations")

# Create random walk features
# We used dim+1 for the RWPE initializion as by default it doesn't return the first dimension of 0's
# That is, in the default setup, dim = 33 would return a 32d vector
dim = 32
initializer = init.RandomWalkPositionalEncodingInitializer(
    triples_factory=dataset.training,
    dim=dim + 1,
)

# build an MLP
hidden = 64
mlp = nn.Sequential(
    nn.Linear(in_features=dim, out_features=hidden),
    nn.ReLU(),
    nn.Linear(in_features=hidden, out_features=dim),
)
r = TransformedRepresentation(
    transformation=mlp,
    # note: this will create an Embedding base representation
    base_kwargs=dict(
        max_id=dataset.num_entities,
        shape=(dim,),
        initializer=initializer,
        trainable=False,
    ),
)
