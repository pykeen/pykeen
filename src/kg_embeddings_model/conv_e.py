import torch
import torch.autograd
import torch.nn as nn
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal

from utilities.constants import NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM, NUM_IN_CHANNELS, NUM_OUT_CHANNELS, \
    KERNEL_HEIGHT, KERNEL_WIDTH, INPUT_DROPOUT, OUTPUT_DROPOUT, FEATURE_MAP_DROPOUT, IMAGE_HEIGHT, IMAGE_WIDTH, \
    BATCH_SIZE, CONV_E

'''
Based on https://github.com/TimDettmers/ConvE/blob/master/model.py
'''


class ConvE(nn.Module):
    def __init__(self, config):
        super(ConvE, self).__init__()
        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.model_name = CONV_E
        num_entities = config[NUM_ENTITIES]
        num_relations = config[NUM_RELATIONS]
        embedding_dim = config[EMBEDDING_DIM]
        num_in_channels = config[NUM_IN_CHANNELS]
        num_out_channels = config[NUM_OUT_CHANNELS]
        kernel_height = config[KERNEL_HEIGHT]
        kernel_width = config[KERNEL_WIDTH]
        input_dropout = config[INPUT_DROPOUT]
        hidden_dropout = config[OUTPUT_DROPOUT]
        feature_map_dropout = config[FEATURE_MAP_DROPOUT]
        self.img_height = config[IMAGE_HEIGHT]
        self.img_width = config[IMAGE_WIDTH]
        self.batch_size = config[BATCH_SIZE]

        assert self.img_height * self.img_width == embedding_dim

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout2d(feature_map_dropout)
        self.loss = torch.nn.BCELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conv1 = torch.nn.Conv2d(in_channels=num_in_channels, out_channels=num_out_channels,
                                     kernel_size=(kernel_height, kernel_width), stride=1, padding=0,
                                     bias=True)

        # num_features – C from an expected input of size (N,C,L)
        self.bn0 = torch.nn.BatchNorm2d(num_in_channels)
        # num_features – C from an expected input of size (N,C,H,W)
        self.bn1 = torch.nn.BatchNorm2d(num_out_channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        num_in_features = num_out_channels * (2 * self.img_height - kernel_height + 1) * (
                    self.img_width - kernel_width + 1)
        self.fc = torch.nn.Linear(num_in_features, embedding_dim)

    def init(self):
        xavier_normal(self.entity_embeddings.weight.data)
        xavier_normal(self.relation_embeddings.weight.data)

    def compute_loss(self, pred, targets):
        """

        :param pred:
        :param targets:
        :return:
        """

        return self.loss(pred,targets)

    def forward(self, e1, rel):
        # batch_size, num_input_channels, width, height
        e1_embedded = self.entity_embeddings(e1).view(-1, 1, self.img_height, self.img_width)
        rel_embedded = self.relation_embeddings(rel).view(-1, 1, self.img_height, self.img_width)

        # batch_size, num_input_channels, 2*height, width
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

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
        x = x.view(self.batch_size, -1)
        x = self.fc(x)

        x = self.hidden_drop(x)

        if self.batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.entity_embeddings.weight.transpose(1, 0))
        # TODO: Why this?
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)

        return pred


import numpy as np
if __name__ == '__main__':
    config = dict()
    config[NUM_ENTITIES] = 8
    config[NUM_RELATIONS] = 2
    config[EMBEDDING_DIM] = 10
    config[NUM_IN_CHANNELS] = 1
    config[NUM_OUT_CHANNELS]= 20
    config[KERNEL_HEIGHT] = 5
    config[KERNEL_WIDTH] = 2
    config[INPUT_DROPOUT] = 0.2
    config[OUTPUT_DROPOUT] = 0.2
    config[FEATURE_MAP_DROPOUT] = 0.2
    config[IMAGE_HEIGHT] = 5
    config[IMAGE_WIDTH] = 2
    config[BATCH_SIZE] = 4

    model = ConvE(config=config)
    subjects = [1,3,5,7]
    subjects = torch.tensor(subjects, dtype=torch.long).view(-1,1)
    objects = [0,2,4,6]
    objects = torch.tensor(objects, dtype=torch.long).view(-1,1)
    relations = [0,0,1,1]
    relations = torch.tensor(relations, dtype=torch.long).view(-1,1)

    steps = 5

    for _ in range(steps):
        scores = model.forward(subjects,relations)
        print(scores)
        print()



