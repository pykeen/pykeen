# -*- coding: utf-8 -*-

"""Implementation of ConvE."""

from typing import Dict

import torch
import torch.autograd
from torch import nn
from torch.nn import Parameter, functional as F
from torch.nn.init import xavier_normal

from pykeen.constants import (
    CONV_E_FEATURE_MAP_DROPOUT, CONV_E_HEIGHT, CONV_E_INPUT_CHANNELS, CONV_E_INPUT_DROPOUT, CONV_E_KERNEL_HEIGHT,
    CONV_E_KERNEL_WIDTH, CONV_E_NAME, CONV_E_OUTPUT_CHANNELS, CONV_E_OUTPUT_DROPOUT, CONV_E_WIDTH, EMBEDDING_DIM,
    NUM_ENTITIES, NUM_RELATIONS,
    MARGIN_LOSS, LEARNING_RATE, PREFERRED_DEVICE, GPU)

__all__ = ['ConvE']


class ConvE(nn.Module):
    """An implementation of ConvE [dettmers2017]_.

    .. [dettmers2017] Dettmers, T., *et al.* (2017) `Convolutional 2d knowledge graph embeddings
                      <https://arxiv.org/pdf/1707.01476.pdf>`_. arXiv preprint arXiv:1707.01476.

    .. seealso:: https://github.com/TimDettmers/ConvE/blob/master/model.py
    """

    model_name = CONV_E_NAME
    hyper_params = [EMBEDDING_DIM, CONV_E_INPUT_CHANNELS, CONV_E_OUTPUT_CHANNELS, CONV_E_HEIGHT, CONV_E_WIDTH,
                    CONV_E_KERNEL_HEIGHT, CONV_E_KERNEL_WIDTH, CONV_E_INPUT_DROPOUT, CONV_E_FEATURE_MAP_DROPOUT,
                    CONV_E_OUTPUT_DROPOUT, MARGIN_LOSS, LEARNING_RATE]

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.try_gpu = config.get(PREFERRED_DEVICE) == GPU

        # Device selection
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and self.try_gpu else 'cpu')

        # Entity dimensions
        self.num_entities = config[NUM_ENTITIES]
        self.num_relations = config[NUM_RELATIONS]

        # Embeddings
        self.embedding_dim = config[EMBEDDING_DIM]

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

        num_in_channels = config[CONV_E_INPUT_CHANNELS]

        num_out_channels = config[CONV_E_OUTPUT_CHANNELS]
        self.img_height = config[CONV_E_HEIGHT]
        self.img_width = config[CONV_E_WIDTH]
        kernel_height = config[CONV_E_KERNEL_HEIGHT]
        kernel_width = config[CONV_E_KERNEL_WIDTH]
        input_dropout = config[CONV_E_INPUT_DROPOUT]
        hidden_dropout = config[CONV_E_OUTPUT_DROPOUT]
        feature_map_dropout = config[CONV_E_FEATURE_MAP_DROPOUT]

        assert self.img_height * self.img_width == self.embedding_dim

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout2d(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_channels,
            out_channels=num_out_channels,
            kernel_size=(kernel_height, kernel_width),
            stride=1,
            padding=0,
            bias=True,
        )

        # num_features – C from an expected input of size (N,C,L)
        self.bn0 = torch.nn.BatchNorm2d(num_in_channels)
        # num_features – C from an expected input of size (N,C,H,W)
        self.bn1 = torch.nn.BatchNorm2d(num_out_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(self.num_entities)))
        num_in_features = num_out_channels * \
                          (2 * self.img_height - kernel_height + 1) * \
                          (self.img_width - kernel_width + 1)
        self.fc = torch.nn.Linear(num_in_features, self.embedding_dim)

    def init(self):  # FIXME is this ever called?
        xavier_normal(self.entity_embeddings.weight.data)
        xavier_normal(self.relation_embeddings.weight.data)

    def predict(self, triples):
        batch_size = triples.shape[0]
        triples = torch.tensor(triples, dtype=torch.long, device=self.device)
        subject_batch = triples[:, 0:1]
        relation_batch = triples[:, 1:2]
        object_batch = triples[:, 2:3].view(-1)

        subject_batch_embedded = self.entity_embeddings(subject_batch).view(-1, 1, self.img_height, self.img_width)
        relation_batch_embedded = self.relation_embeddings(relation_batch).view(-1, 1, self.img_height, self.img_width)
        candidate_object_emebddings = self.entity_embeddings(object_batch)

        # batch_size, num_input_channels, 2*height, width
        stacked_inputs = torch.cat([subject_batch_embedded, relation_batch_embedded], 2)

        # batch_size, num_input_channels, 2*height, width
        stacked_inputs = self.bn0(stacked_inputs)

        # batch_size, num_input_channels, 2*height, width
        x = self.inp_drop(stacked_inputs)
        # (N,C_out,H_out,W_out)
        x = self.conv1(x)

        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        # batch_size, num_output_channels * (2 * height - kernel_height + 1) * (width - kernel_width + 1)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)

        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, candidate_object_emebddings.transpose(1, 0))
        scores = F.sigmoid(x)

        max_scores, predictions = torch.max(scores, dim=1)

        # Class 0 represents false fact and class 1 represents true fact

        return predictions

    def forward(self, batch, labels):
        batch_size = batch.shape[0]

        heads = batch[:, 0:1]
        relations = batch[:, 1:2]
        tails = batch[:, 2:3]

        # batch_size, num_input_channels, width, height
        heads_embs = self.entity_embeddings(heads).view(-1, 1, self.img_height, self.img_width)
        relation_embs = self.relation_embeddings(relations).view(-1, 1, self.img_height, self.img_width)
        tails_embs = self.entity_embeddings(tails).view(-1, self.embedding_dim)

        # batch_size, num_input_channels, 2*height, width
        stacked_inputs = torch.cat([heads_embs, relation_embs], 2)

        # batch_size, num_input_channels, 2*height, width
        stacked_inputs = self.bn0(stacked_inputs)

        # batch_size, num_input_channels, 2*height, width
        x = self.inp_drop(stacked_inputs)
        # (N,C_out,H_out,W_out)
        x = self.conv1(x)

        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        # batch_size, num_output_channels * (2 * height - kernel_height + 1) * (width - kernel_width + 1)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)

        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)

        scores = torch.sum(torch.mm(x, tails_embs.transpose(1, 0)), dim=1)

        predictions = F.sigmoid(scores)
        loss = self.loss(predictions, labels)
        return loss
