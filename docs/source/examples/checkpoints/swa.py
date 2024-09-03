"""
Stochastic weight averaging.

See also:
    - https://pytorch.org/docs/stable/optim.html#weight-averaging-swa-and-ema
"""

# %%
import pathlib
import tempfile

from pykeen.checkpoints.average import exponential_moving_weight_average, stochastic_weight_average
from pykeen.datasets.utils import get_dataset
from pykeen.evaluation import evaluator_resolver
from pykeen.models import model_resolver
from pykeen.pipeline import pipeline
from pykeen.trackers.base import PythonResultTracker

dataset = get_dataset(dataset="codexsmall")
model = model_resolver.make("MuRE", triples_factory=dataset.training)
evaluator = evaluator_resolver.make(None)
checkpoint_root = pathlib.Path(tempfile.gettempdir()).joinpath("checkpoints")
for p in checkpoint_root.glob("*.pt"):
    p.unlink()
result_tracker = PythonResultTracker()
evaluation_triples = dataset.validation.mapped_triples
evaluation_kwargs = dict(
    additional_filter_triples=dataset.training.mapped_triples,
    batch_size=256,
)

# train model and store checkpoints
result = pipeline(
    dataset=dataset,
    model=model,
    training_kwargs=dict(
        num_epochs=100,
        callbacks=["checkpoint", "evaluation"],
        callbacks_kwargs=[
            dict(
                schedule="every",
                schedule_kwargs=dict(frequency=1),
                keeper="last",
                keeper_kwargs=dict(keep=5),
                root=checkpoint_root,
            ),
            dict(
                evaluation_triples=evaluation_triples,
                **evaluation_kwargs,
                prefix="validation",
            ),
        ],
    ),
    result_tracker=result_tracker,
    evaluator=evaluator,
    use_testing_data=False,
)

# %%
results = evaluator.evaluate(model, mapped_triples=evaluation_triples, **evaluation_kwargs)
# %%
checkpoints = sorted(checkpoint_root.glob("checkpoint*.pt"))
swa_model = stochastic_weight_average(model, checkpoints=checkpoints)
swa_results = evaluator.evaluate(swa_model, mapped_triples=evaluation_triples, **evaluation_kwargs)
# %%
ema_model = exponential_moving_weight_average(model, checkpoints=checkpoints, decay=0.1)
ema_results = evaluator.evaluate(ema_model, mapped_triples=evaluation_triples, **evaluation_kwargs)

# %%
