import torch
import torch.autograd
import torch.nn as nn
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal, xavier_uniform
from utilities.constants import NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM, NUM_IN_CHANNELS, NUM_OUT_CHANNELS, \
    KERNEL_HEIGHT, KERNEL_WIDTH, INPUT_DROPOUT, OUTPUT_DROPOUT, FEATURE_MAP_DROPOUT

'''
Based on https://github.com/TimDettmers/ConvE/blob/master/model.py
'''


class ConvE(nn.Module):
    def __init__(self, config):
        super(ConvE, self).__init__()
        # A simple lookup table that stores embeddings of a fixed dictionary and size

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
        # TODO: Check wheter feature_size makes sense
        self.bn0 = torch.nn.BatchNorm2d(num_in_channels)
        self.bn1 = torch.nn.BatchNorm2d(num_out_channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(10368, embedding_dim)


    def init(self):
        xavier_normal(self.entity_embeddings.weight.data)
        xavier_normal(self.relation_embeddings.weight.data)

    def forward(self, e1, rel):
        batch_size = 1
        e1_embedded = self.entity_embeddings(e1).view(-1, 1, 10, 20)
        rel_embedded = self.relation_embeddings(rel).view(-1, 1, 10, 20)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        # print(x.size())
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.entity_embeddings.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)

        return pred

if __name__ == '__main__':
    config = {}
    config[NUM_ENTITIES] = 2
    config[NUM_RELATIONS] = 1
    config[EMBEDDING_DIM] = 5
    config[NUM_IN_CHANNELS] = 1
    config[NUM_OUT_CHANNELS] = 3
    config[KERNEL_HEIGHT] = 2
    config[KERNEL_WIDTH] = 2
    config[INPUT_DROPOUT] = 0.5
    config[OUTPUT_DROPOUT] = 0.5
    config[FEATURE_MAP_DROPOUT] = 0.5
    ConvE(config=config)