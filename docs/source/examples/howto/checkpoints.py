"""Saving checkpoints during training."""

# %%
from pykeen.pipeline import pipeline

pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    optimizer="Adam",
    training_kwargs={
        "num_epochs": 1000,
    },
)

# %%
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    optimizer="Adam",
    training_kwargs={
        "num_epochs": 1000,
        "checkpoint_name": "my_checkpoint.pt",
    },
)

# %%
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    optimizer="Adam",
    training_kwargs={
        "num_epochs": 1000,
        "checkpoint_name": "my_checkpoint.pt",
        "checkpoint_frequency": 5,
    },
)

# %%
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    optimizer="Adam",
    training_kwargs={
        "num_epochs": 2000,  # more epochs than before
        "checkpoint_name": "my_checkpoint.pt",
        "checkpoint_frequency": 5,
    },
)

# %%
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    optimizer="Adam",
    training_kwargs={
        "num_epochs": 2000,
        "checkpoint_name": "my_checkpoint.pt",
        "checkpoint_directory": "doctests/checkpoint_dir",
    },
)

# %%
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    optimizer="Adam",
    training_kwargs={
        "num_epochs": 2000,
        "checkpoint_on_failure": True,
    },
)

# %%
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    optimizer="Adam",
    training_kwargs={
        "num_epochs": 2000,
        "checkpoint_name": "my_checkpoint.pt",
        "checkpoint_on_failure": True,
    },
)

# %%
from pykeen.datasets.nations import NATIONS_TEST_PATH, NATIONS_TRAIN_PATH, NATIONS_VALIDATE_PATH
from pykeen.triples import TriplesFactory

training = TriplesFactory.from_path(
    path=NATIONS_TRAIN_PATH,
)
validation = TriplesFactory.from_path(
    path=NATIONS_VALIDATE_PATH,
    entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,
)
testing = TriplesFactory.from_path(
    path=NATIONS_TEST_PATH,
    entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,
)
pipeline_result = pipeline(
    training=training,
    validation=validation,
    testing=testing,
    model="TransE",
    optimizer="Adam",
    training_kwargs={
        "num_epochs": 2000,
        "checkpoint_name": "my_checkpoint.pt",
    },
)

# %%
import torch

from pykeen.constants import PYKEEN_CHECKPOINTS

checkpoint = torch.load(PYKEEN_CHECKPOINTS.joinpath("my_checkpoint.pt"))

# %%
training = TriplesFactory.from_path(
    path=NATIONS_TRAIN_PATH,
    entity_to_id=checkpoint["entity_to_id_dict"],
    relation_to_id=checkpoint["relation_to_id_dict"],
)
validation = TriplesFactory.from_path(
    path=NATIONS_VALIDATE_PATH,
    entity_to_id=checkpoint["entity_to_id_dict"],
    relation_to_id=checkpoint["relation_to_id_dict"],
)
testing = TriplesFactory.from_path(
    path=NATIONS_TEST_PATH,
    entity_to_id=checkpoint["entity_to_id_dict"],
    relation_to_id=checkpoint["relation_to_id_dict"],
)

# %%
pipeline_result = pipeline(
    training=training,
    validation=validation,
    testing=testing,
    model="TransE",
    optimizer="Adam",
    training_kwargs={
        "num_epochs": 2000,
        "checkpoint_name": "my_checkpoint.pt",
    },
)

# %%
checkpoint = torch.load(PYKEEN_CHECKPOINTS.joinpath("my_checkpoint.pt"))

# %%
train = TriplesFactory.from_path(
    path=NATIONS_TRAIN_PATH,
    entity_to_id=checkpoint["entity_to_id_dict"],
    relation_to_id=checkpoint["relation_to_id_dict"],
)

# %%
from pykeen.models import TransE

my_model = TransE(triples_factory=train)
my_model.load_state_dict(checkpoint["model_state_dict"])

# %%
from torch.optim import Adam

from pykeen.datasets import Nations
from pykeen.training import SLCWATrainingLoop

triples_factory = Nations().training
model = TransE(
    triples_factory=triples_factory,
    random_seed=123,
)
optimizer = Adam(params=model.get_grad_params())
training_loop = SLCWATrainingLoop(model=model, optimizer=optimizer)

# %%
losses = training_loop.train(
    num_epochs=1000,
    checkpoint_name="my_checkpoint.pt",
    checkpoint_frequency=5,
)

# %%
losses = training_loop.train(
    num_epochs=2000,
    checkpoint_name="my_checkpoint.pt",
    checkpoint_frequency=5,
)

# %%
losses = training_loop.train(
    num_epochs=2000,
    checkpoint_directory="/my/secret/dir",
    checkpoint_on_failure=True,
)
