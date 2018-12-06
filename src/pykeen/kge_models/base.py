# -*- coding: utf-8 -*-

"""Utilities for getting and initializing KGE models."""

import torch
from torch import nn

from pykeen.constants import CPU, GPU, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS, PREFERRED_DEVICE

__all__ = [
    'BaseModule',
]


class BaseModule(nn.Module):
    """A base class for all of the models."""

    #: Should the margin ranking loss use the size average?
    margin_ranking_loss_size_average: bool = ...

    def __init__(self, config):
        super().__init__()

        # Device selection
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and config[PREFERRED_DEVICE] == GPU else CPU)

        # Loss
        self.margin_loss = config[MARGIN_LOSS]
        self.criterion = nn.MarginRankingLoss(
            margin=self.margin_loss,
            size_average=self.margin_ranking_loss_size_average,
        )

        # Entity dimensions
        self.num_entities = config[NUM_ENTITIES]
        self.num_relations = config[NUM_RELATIONS]
