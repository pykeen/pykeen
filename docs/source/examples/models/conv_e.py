"""Example of using ConvE outside of the pipeline."""

# Step 1: Get triples
from pykeen.datasets import get_dataset

dataset = get_dataset(dataset="nations", dataset_kwargs=dict(create_inverse_triples=True))

# Step 2: Configure the model
from pykeen.models import ConvE

model = ConvE(
    triples_factory=dataset.training,
    embedding_dim=200,
    input_channels=1,
    output_channels=32,
    embedding_height=10,
    embedding_width=20,
    kernel_height=3,
    kernel_width=3,
    input_dropout=0.2,
    feature_map_dropout=0.2,
    output_dropout=0.3,
)

# Step 3: Configure the loop
from torch.optim import Adam

optimizer = Adam(params=model.get_grad_params())
from pykeen.training import LCWATrainingLoop

training_loop = LCWATrainingLoop(model=model, optimizer=optimizer)

# Step 4: Train
losses = training_loop.train(triples_factory=dataset.training, num_epochs=5, batch_size=256)

# Step 5: Evaluate the model
from pykeen.evaluation import RankBasedEvaluator

evaluator = RankBasedEvaluator()
metric_result = evaluator.evaluate(
    model=model,
    mapped_triples=dataset.testing.mapped_triples,
    additional_filter_triples=dataset.training.mapped_triples,
    batch_size=8192,
)
