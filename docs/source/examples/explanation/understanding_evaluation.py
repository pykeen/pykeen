"""Understanding the evaluation."""

# %%
# As an example, consider we trained a KGEM on the countries dataset.
from pykeen.datasets import get_dataset
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

dataset = get_dataset(dataset="countries")
result = pipeline(dataset=dataset, model="mure", random_seed=42, training_kwargs={"num_epochs": 100})

# %%
# During evaluation time, we now evaluate head and tail prediction, i.e., whether we can predict the correct
# head/tail entity from the remainder of a triple. The first triple in the test split of this dataset is
# ``['belgium', 'locatedin', 'europe']``. Thus, for tail prediction, we aim to answer ``['belgium', 'locatedin', ?]``.
# We can see the results using the prediction workflow:
from pykeen.predict import predict_target

df = predict_target(
    model=result.model,
    head="belgium",
    relation="locatedin",
    triples_factory=result.training,
)

# %%
# As an example, we can inspect the :class:`~pykeen.datasets.WD50KT` dataset, where a single (relation,
# tail)-combination, ("instance of", "human"), is present in 699 evaluation triples.
dataset = get_dataset(dataset="wd50kt")
testing = dataset.testing
assert isinstance(testing, TriplesFactory)
unique_relation_tail, counts = testing.mapped_triples[:, 1:].unique(return_counts=True, dim=0)
c = counts.max()  # c = 699
r, t = unique_relation_tail[counts.argmax()]
t = testing.entity_id_to_label[t.item()]  # https://www.wikidata.org/wiki/Q5 -> "human"
r = testing.relation_id_to_label[r.item()]  # https://www.wikidata.org/wiki/Property:P31 -> "instance of"

# %%
# With the [bordes2013]_ technique where the testing set is used for evaluation, the ``additional_filter_triples``
# should include both the training triples and validation triples as in the following example:
from pykeen.datasets import FB15k237
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import TransE

# Get FB15k-237 dataset
fb15k237 = FB15k237()
assert isinstance(fb15k237.validation, TriplesFactory)

# Define model
model = TransE(
    triples_factory=fb15k237.training,
)

# Train your model (code is omitted for brevity)

# Define evaluator
evaluator = RankBasedEvaluator(
    filtered=True,  # Note: this is True by default; we're just being explicit
)

# Evaluate your model with not only testing triples,
# but also filter on validation triples
results = evaluator.evaluate(
    model=model,
    mapped_triples=fb15k237.testing.mapped_triples,
    additional_filter_triples=[
        fb15k237.training.mapped_triples,
        fb15k237.validation.mapped_triples,
    ],
)

# %%
# The edges in the Hetionet graph are listed `here
# <https://github.com/hetio/hetionet/blob/master/describe/edges/metaedges.tsv>`_, but we will focus on only the
# compound treat disease (CtD) and compound palliates disease (CpD) relations during evaluation. This can be done
# with the following:
evaluation_relation_whitelist = {"CtD", "CpD"}
pipeline_result = pipeline(
    dataset="Hetionet",
    model="RotatE",
    evaluation_relation_whitelist=evaluation_relation_whitelist,
)

# %%
# The HPO pipeline accepts the same arguments:
from pykeen.hpo import hpo_pipeline

hpo_pipeline_result = hpo_pipeline(
    n_trials=30,
    dataset="Hetionet",
    model="RotatE",
    evaluation_relation_whitelist=evaluation_relation_whitelist,
)
