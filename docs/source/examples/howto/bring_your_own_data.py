"""Bring your own data to the pipeline."""

# %%
from pykeen.datasets.nations import NATIONS_TEST_PATH, NATIONS_TRAIN_PATH, NATIONS_VALIDATE_PATH
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

result = pipeline(
    training=NATIONS_TRAIN_PATH,
    testing=NATIONS_TEST_PATH,
    model="TransE",
    epochs=5,  # short epochs for testing - you should go higher
)
result.save_to_directory("doctests/test_pre_stratified_transe")

# %%
from pykeen.hpo import hpo_pipeline

hpo_pipeline_result = hpo_pipeline(
    n_trials=3,  # you probably want more than this
    training=NATIONS_TRAIN_PATH,
    testing=NATIONS_TEST_PATH,
    validation=NATIONS_VALIDATE_PATH,
    model="TransE",
    epochs=5,  # short epochs for testing - you should go higher
)
hpo_pipeline_result.save_to_directory("doctests/test_hpo_pre_stratified_transe")

# %%
result = pipeline(
    training=NATIONS_TRAIN_PATH,
    testing=NATIONS_TEST_PATH,
    dataset_kwargs={"create_inverse_triples": True},
    model="TransE",
    epochs=5,  # short epochs for testing - you should go higher
)
result.save_to_directory("doctests/test_pre_stratified_transe")

# %%
training = TriplesFactory.from_path(NATIONS_TRAIN_PATH)
testing = TriplesFactory.from_path(
    NATIONS_TEST_PATH,
    entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,
)
result = pipeline(
    training=training,
    testing=testing,
    model="TransE",
    epochs=5,  # short epochs for testing - you should go higher
)
result.save_to_directory("doctests/test_pre_stratified_transe")

# %%
training = TriplesFactory.from_path(
    NATIONS_TRAIN_PATH,
    create_inverse_triples=True,
)
testing = TriplesFactory.from_path(
    NATIONS_TEST_PATH,
    entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,
    create_inverse_triples=True,
)
result = pipeline(
    training=training,
    testing=testing,
    model="TransE",
    epochs=5,  # short epochs for testing - you should go higher
)
result.save_to_directory("doctests/test_pre_stratified_transe")

# %%
tf = TriplesFactory.from_path(NATIONS_TRAIN_PATH)
training, testing = tf.split()
result = pipeline(
    training=training,
    testing=testing,
    model="TransE",
    epochs=5,  # short epochs for testing - you should go higher
)
result.save_to_directory("doctests/test_unstratified_transe")

# %%
tf = TriplesFactory.from_path(NATIONS_TRAIN_PATH)
training, testing, validation = tf.split([0.8, 0.1, 0.1])
result = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model="TransE",
    stopper="early",
    epochs=5,  # short epochs for testing - you should go
    # higher, especially with early stopper enabled
)
result.save_to_directory("doctests/test_unstratified_stopped_transe")
