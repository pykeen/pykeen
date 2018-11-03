# -*- coding: utf-8 -*-

'''Script for predicting links based on a trained model.'''
import json
import os

import numpy as np
import torch
from pykeen.constants import PREFERRED_DEVICE, GPU, CPU
from pykeen.utilities.initialization_utils.module_initialization_utils import get_kg_embedding_model
from pykeen.utilities.prediction_utils import make_predictions


def start_predictions_piepline(model_direc, data_direc):
    # Load configuration file
    in_path = os.path.join(model_direc, 'configuration.json')
    with open(in_path) as f:
        config = json.load(f)

    # Load entity to id mapping
    in_path = os.path.join(model_direc, 'entity_to_id.json')
    with open(in_path) as f:
        entity_to_id = json.load(f)

    # Load relation to id mapping
    in_path = os.path.join(model_direc, 'relation_to_id.json')
    with open(in_path) as f:
        relation_to_id = json.load(f)

    trained_model = get_kg_embedding_model(config=config)
    path_to_model = os.path.join(model_direc, 'trained_model.pkl')
    trained_model.load_state_dict(torch.load(path_to_model))

    in_path = os.path.join(data_direc, 'entities.tsv')
    entities = np.loadtxt(fname=in_path, dtype=str)

    in_path = os.path.join(data_direc, 'relations.tsv')
    relations = np.loadtxt(fname=in_path, dtype=str)

    device_name = 'cuda:0' if torch.cuda.is_available() and config[PREFERRED_DEVICE] == GPU else CPU

    device = torch.device(device_name)

    ranked_triples = make_predictions(kg_model=trained_model,
                                      entities=entities,
                                      relations=relations,
                                      entity_to_id=entity_to_id,
                                      rel_to_id=relation_to_id,
                                      device=device)

    out_path = os.path.join(data_direc, 'predictions.tsv')
    np.savetxt(out_path, ranked_triples, fmt='%s')
