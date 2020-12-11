"""Invese Stability Workflow."""

import matplotlib.pyplot as plt
import seaborn as sns

from pykeen.constants import PYKEEN_EXPERIMENTS
from pykeen.datasets import get_dataset
from pykeen.pipeline import pipeline

INVERSE_STABILITY = PYKEEN_EXPERIMENTS / 'inverse_stability'
INVERSE_STABILITY.mkdir(parents=True, exist_ok=True)


def run_inverse_stability_workflow(dataset, model: str, ):
    dataset = get_dataset(dataset=dataset)
    pipeline_result = pipeline(
        dataset=dataset,
        dataset_kwargs=dict(
            create_inverse_triples=True,
        ),
        model=model,
        model_kwargs=dict(
            embedding_dim=50,
        ),
    )
    test_tf = dataset.testing
    model = pipeline_result.model
    # Score with original triples
    scores = model.score_hrt(test_tf.mapped_triples)

    # Score with inverse triples
    scores_inverse = model.score_hrt_inverse(test_tf.mapped_triples)

    fig, ax = plt.subplots(1, 1)
    sns.distplot(scores, ax=ax)
    sns.distplot(scores_inverse, ax=ax)
    plt.savefig(INVERSE_STABILITY / 'results.png', dpi=300)
