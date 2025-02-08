"""Creating a fully inductive dataset."""

import logging

from pykeen.datasets import get_dataset
from pykeen.datasets.inductive.base import EagerInductiveDataset

logging.basicConfig(level=logging.INFO)

# we use all of CodexSmall's data as source graph
dataset = get_dataset(dataset="CodexSmall")
dataset.summarize()
tf_all = dataset.merged()

# create a fully inductive split with two evaluation parts (validation & test)
tf_training, tf_inference, tf_validation, tf_testing = tf_all.split_fully_inductive(
    entity_split_train_ratio=0.5, evaluation_triples_ratios=(0.8, 0.1), random_state=42
)

inductive_dataset = EagerInductiveDataset(
    transductive_training=tf_training,
    inductive_inference=tf_inference,
    inductive_testing=tf_testing,
    inductive_validation=tf_validation,
    create_inverse_triples=False,
)
inductive_dataset.summarize()
