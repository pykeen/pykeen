import numpy as np
import torch

import pykeen.losses
from pykeen.datasets import WN18RR
from pykeen.models.unimodal.boxe_kg import BoxEKG
from pykeen.pipeline import pipeline


# TODO: Align optimizations: NSSALoss is printing invalid values here, but is sensible in the forward loop
# TODO: Align optimizer settings: Constant LR


def main():
    embedding_dim = 500
    unif_init_bound = 2 * np.sqrt(embedding_dim)
    init_kw = dict(a=-1 / unif_init_bound, b=1 / unif_init_bound)
    size_init_kw = dict(a=-1, b=1)
    dataset = WN18RR()
    triples_factory = dataset.training
    model = BoxEKG(
        triples_factory=triples_factory,
        embedding_dim=500,
        norm_order=2,
        tanh_map=True,
        entity_initializer_kwargs=init_kw,
        relation_initializer_kwargs=init_kw,
        relation_size_initializer_kwargs=size_init_kw,
    )

    results = pipeline(
        random_seed=1000000,
        dataset=dataset,
        model=model,
        training_kwargs=dict(num_epochs=300, batch_size=512, checkpoint_name="trial.pt", checkpoint_frequency=100),
        loss=pykeen.losses.NSSALoss(margin=3, adversarial_temperature=2.0, reduction="sum"),
        training_loop="sLCWA",
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=150),
        result_tracker="json",
        result_tracker_kwargs=dict(name="test.json"),
        evaluation_kwargs=dict(batch_size=16),
        optimizer=torch.optim.Adam,
    )


if __name__ == '__main__':
    main()
