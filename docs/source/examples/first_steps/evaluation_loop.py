"""Preview: Evaluation Loop."""

from pykeen.datasets import Nations
from pykeen.evaluation import LCWAEvaluationLoop
from pykeen.models import TransE
from pykeen.training import SLCWATrainingLoop

# get a dataset
dataset = Nations()
# Pick a model
model = TransE(triples_factory=dataset.training)
# Pick a training approach (sLCWA or LCWA)
training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=dataset.training,
)
# Train like Cristiano Ronaldo
_ = training_loop.train(
    triples_factory=dataset.training,
    num_epochs=5,
    batch_size=256,
    # NEW: validation evaluation callback
    callbacks="evaluation-loop",
    callbacks_kwargs=dict(
        prefix="validation",
        factory=dataset.validation,
    ),
)
# Pick an evaluation loop (NEW)
evaluation_loop = LCWAEvaluationLoop(
    model=model,
    triples_factory=dataset.testing,
)
# Evaluate
results = evaluation_loop.evaluate()
# print(results)
