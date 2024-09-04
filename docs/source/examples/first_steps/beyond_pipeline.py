"""Beyond the pipeline."""

# Get a training dataset
from pykeen.datasets import get_dataset

dataset = get_dataset(dataset="nations")
training = dataset.training
validation = dataset.validation
testing = dataset.testing
# The following applies to most packaged datasets,
# although the dataset class itself makes `validation' optional.
assert validation is not None


# Pick a model
from pykeen.models import TransE

model = TransE(triples_factory=training)


# Pick an optimizer from PyTorch
from torch.optim import Adam

optimizer = Adam(params=model.get_grad_params())


# Pick a training approach (sLCWA or LCWA)
from pykeen.training import SLCWATrainingLoop

training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training,
    optimizer=optimizer,
)


# Train like Cristiano Ronaldo
_ = training_loop.train(
    triples_factory=training,
    num_epochs=5,
    batch_size=256,
)


# Pick an evaluator
from pykeen.evaluation import RankBasedEvaluator

evaluator = RankBasedEvaluator()


# Evaluate
results = evaluator.evaluate(
    model=model,
    mapped_triples=testing.mapped_triples,
    batch_size=1024,
    additional_filter_triples=[
        training.mapped_triples,
        validation.mapped_triples,
    ],
)

# print(results)
