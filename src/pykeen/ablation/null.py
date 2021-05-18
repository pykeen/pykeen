# -*- coding: utf-8 -*-

"""Produce a randomized ablation study for testing purposes."""

from typing import Optional, TypeVar, cast

import click
import torch
from more_click import verbose_option
from pykeen.ablation import ablation_pipeline
from pykeen.constants import PYKEEN_EXPERIMENTS
from pykeen.losses import Loss, loss_resolver
from pykeen.models import TransE, model_resolver
from torch import FloatTensor
from torch.nn.init import uniform_

from class_resolver import Resolver

X = TypeVar('X')


class RandomModel(TransE):
    """Generate random scores."""

    def forward(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
        slice_size: Optional[int] = None,
        slice_dim: Optional[str] = None,
    ) -> FloatTensor:
        """Return a random result."""
        # shape: (batch_size, num_heads, num_relations, num_tails)
        size = 1, self.num_entities, self.num_relations, self.num_relations  # TODO
        rv = torch.empty(size=size)
        uniform_(rv)
        return cast(FloatTensor, rv)


class RandomLoss(Loss):
    """Generate random losses."""

    def forward(
        self,
        scores: FloatTensor,
        labels: FloatTensor,
    ) -> FloatTensor:
        """Return a random result."""
        with torch.no_grad():
            rv = cast(FloatTensor, torch.rand(1))
        return rv


def mock_resolver(resolver: Resolver, cls):
    resolver.register(cls)
    for key in resolver.lookup_dict:
        resolver.lookup_dict[key] = cls
    for synonym in resolver.synonyms or []:
        resolver.synonyms[synonym] = cls


mock_resolver(model_resolver, RandomModel)
mock_resolver(loss_resolver, RandomLoss)


def random_ablation():
    ablation_pipeline(
        datasets='Nations',
        models=[
            'TransE',
            'TransR',
        ],
        losses=[
            'NSSA',
            'BCE',
            'MR',
        ],
        directory=PYKEEN_EXPERIMENTS.joinpath('rand_ablation'),
        optimizers='Adam',
        training_loops=['LCWA', 'sLCWA'],
    )


@click.command()
@verbose_option
def main():
    random_ablation()


if __name__ == '__main__':
    main()
