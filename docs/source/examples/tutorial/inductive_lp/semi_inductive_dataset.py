"""Creating a semi-inductive dataset."""

import logging

import torch

from pykeen.datasets import get_dataset
from pykeen.triples.splitting import split_semi_inductive
from pykeen.triples.triples_factory import TriplesFactory

logging.basicConfig(level=logging.INFO)

# we use all of CodexSmall's data as source graph
dataset = get_dataset(dataset="CodexSmall")
mapped_triples = torch.cat([tf.mapped_triples for tf in dataset.factory_dict.values()])

# create a fully inductive split with two evaluation parts (validation & test)
training, validation, test = split_semi_inductive(
    mapped_triples=mapped_triples,
    ratios=(0.8, 0.1),
    random_state=42,
)
tf_training, tf_validation, tf_testing = (
    TriplesFactory(
        mapped_triples=mapped_triples, entity_to_id=dataset.entity_to_id, relation_to_id=dataset.relation_to_id
    )
    for mapped_triples in (training, validation, test)
)
print(tf_training, tf_validation, tf_testing)
