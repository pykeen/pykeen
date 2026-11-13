"""Train TransE with different learning rates for entity and relation embeddings."""

from torch.optim import Adam

from pykeen.datasets import Nations
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import TransE
from pykeen.training import SLCWATrainingLoop

# Nations is packaged with PyKEEN, so no dataset download is needed.
dataset = Nations()
model = TransE(triples_factory=dataset.training, embedding_dim=32, random_seed=42).to("cpu")

# TransE has one entity representation and one relation representation.
# Give each parameter to exactly one optimizer group.
optimizer = Adam(
    [
        {"params": model.entity_representations.parameters(), "lr": 1.0e-4},
        {"params": model.relation_representations.parameters(), "lr": 1.0e-3},
    ]
)

training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=dataset.training,
    optimizer=optimizer,
)
losses = training_loop.train(
    triples_factory=dataset.training,
    num_epochs=5,
    batch_size=256,
    use_tqdm=False,
    use_tqdm_batch=False,
)

# Inspect the optimizer groups after training.
learning_rates = [group["lr"] for group in optimizer.param_groups]

# Filter other known true triples when evaluating on the test split.
assert dataset.validation is not None
results = RankBasedEvaluator().evaluate(
    model=model,
    mapped_triples=dataset.testing.mapped_triples,
    additional_filter_triples=[
        dataset.training.mapped_triples,
        dataset.validation.mapped_triples,
    ],
    batch_size=256,
    use_tqdm=False,
)
