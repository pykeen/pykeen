"""An example for exchanging the triples factory after training."""

import logging

from pykeen.datasets import get_dataset
from pykeen.models import InductiveNodePiece
from pykeen.pipeline import pipeline
from pykeen.predict import predict_triples
from pykeen.triples.generation import generate_triples_factory

logging.basicConfig(level=logging.INFO)

# we use all of CodexSmall's data as source graph
dataset = get_dataset(dataset="CodexSmall")
dataset.summarize()
tf_all = dataset.merged()

# create a fully inductive split with two evaluation parts (validation & test)
tf_training, tf_inference, tf_validation, tf_testing = tf_all.split_fully_inductive(
    entity_split_train_ratio=0.5, evaluation_triples_ratios=(0.8, 0.1), random_state=42
)

# train an inductive node piece model
tf_training.create_inverse_triples = True
result = pipeline(
    training=tf_training,
    testing=tf_testing,
    dataset_kwargs=dict(create_inverse_triples=True),
    model="InductiveNodePiece",
    model_kwargs=dict(inference_factory=tf_inference),
    training_kwargs=dict(num_epochs=0),
    training_loop_kwargs=dict(mode="training"),
    evaluator_kwargs=dict(mode="validation"),
)

# inference some validation triples
df = predict_triples(model=result.model, triples_factory=tf_validation, mode="validation").process()
print(df)

# we replace the validation factory by a faked triples factory (which we mock here with random triples)
# note: we need to keep the relations fixed (because of NodePiece's parametrization)!
tf_new = generate_triples_factory(
    num_entities=13,
    num_relations=tf_training.real_num_relations,
    random_state=42,
    create_inverse_triples=False,
)
model: InductiveNodePiece = result.model
model.replace_entity_representations_(
    mode="validation", representation=model.create_entity_representation_for_new_triples(tf_new)
)

# inference new validation triples
df = predict_triples(model=result.model, triples_factory=tf_new, mode="validation").process()
print(df)
