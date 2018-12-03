# -*- coding: utf-8 -*-

"""Script for predicting links based on a trained model."""

import json
import os

import numpy as np
import torch
from torch.nn import Module

from pykeen.constants import CPU, GPU, PREFERRED_DEVICE
from pykeen.kge_models import get_kge_model
from pykeen.utilities.prediction_utils import make_predictions


def start_predictions_pipeline(model_direc: str, data_direc: str):
    # Load configuration file
    with open(os.path.join(model_direc, 'configuration.json')) as f:
        config = json.load(f)

    # Load entity to id mapping
    with open(os.path.join(model_direc, 'entity_to_id.json')) as f:
        entity_to_id = json.load(f)

    # Load relation to id mapping
    with open(os.path.join(model_direc, 'relation_to_id.json')) as f:
        relation_to_id = json.load(f)

    trained_kge_model: Module = get_kge_model(config=config)
    path_to_model = os.path.join(model_direc, 'trained_model.pkl')
    trained_kge_model.load_state_dict(torch.load(path_to_model))

    entities = np.loadtxt(fname=os.path.join(data_direc, 'entities.tsv'), dtype=str, delimiter='\t')
    relations = np.loadtxt(fname=os.path.join(data_direc, 'relations.tsv'), dtype=str, delimiter='\t')

    device_name = 'cuda:0' if torch.cuda.is_available() and config[PREFERRED_DEVICE] == GPU else CPU
    device = torch.device(device_name)

    ranked_triples = make_predictions(
        kge_model=trained_kge_model,
        entities=entities,
        relations=relations,
        entity_to_id=entity_to_id,
        rel_to_id=relation_to_id,
        device=device,
    )

    np.savetxt(os.path.join(data_direc, 'predictions.tsv'), ranked_triples, fmt='%s')
