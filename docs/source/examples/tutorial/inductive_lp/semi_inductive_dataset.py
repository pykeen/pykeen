"""Creating a semi-inductive dataset."""

import logging

from pykeen.datasets import get_dataset
from pykeen.datasets.base import EagerDataset

logging.basicConfig(level=logging.INFO)

# we use all of CodexSmall's data as source graph
dataset = get_dataset(dataset="CodexSmall")
dataset.summarize()
tf_all = dataset.merged()

# create a fully inductive split with two evaluation parts (validation & test)
tf_training, tf_validation, tf_testing = tf_all.split_semi_inductive(ratios=(0.8, 0.1), random_state=42)
dataset = EagerDataset(training=tf_training, testing=tf_testing, validation=tf_validation)
dataset.summarize()
