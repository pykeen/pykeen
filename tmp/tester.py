import numpy as np
import torch

from pykeen.losses import NSSALoss
from pykeen.datasets import WN18RR
from pykeen.models.unimodal.boxe import BoxE
from pykeen.pipeline import pipeline


def main():
    embedding_dim = 500
    unif_init_bound = 2 * np.sqrt(embedding_dim)
    init_kw = dict(a=-1 / unif_init_bound, b=1 / unif_init_bound)
    size_init_kw = dict(a=-1, b=1)
    model_keyword_args = dict(
        embedding_dim=500,
        # norm_order=2,
        p=2,
        power_norm=False,
        tanh_map=True,
        entity_initializer=torch.nn.init.uniform_,
        entity_initializer_kwargs=init_kw,
        relation_initializer=torch.nn.init.uniform_,
        relation_initializer_kwargs=init_kw,
        relation_size_initializer=torch.nn.init.uniform_,
        relation_size_initializer_kwargs=size_init_kw,
    )

    results = pipeline(
        random_seed=1000000,
        dataset="WN18RR",
        model=BoxE,
        model_kwargs=model_keyword_args,
        training_kwargs=dict(num_epochs=300, batch_size=512, checkpoint_name="triam.pt", checkpoint_frequency=100),
        loss=NSSALoss(margin=5, adversarial_temperature=0.0, reduction="sum"),
        training_loop="sLCWA",
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=150),
        evaluation_kwargs=dict(batch_size=16),
        optimizer=torch.optim.Adam,
        optimizer_kwargs=dict(lr=0.001),
    )


if __name__ == "__main__":
    main()
