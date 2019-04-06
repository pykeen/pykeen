# -*- coding: utf-8 -*-

"""Script for predicting links based on a trained model."""

import json
import os
from typing import Optional

import numpy as np
import torch
from torch.nn import Module

from pykeen.constants import CPU, GPU, PREFERRED_DEVICE
from pykeen.kge_models import get_kge_model
from pykeen.utilities.prediction_utils import make_predictions


def start_predictions_pipeline(model_directory: str,
                               data_directory: str,
                               path_to_blacklisted_triples: Optional[str] = None,
                               export_predictions=True) -> None:
    """
    Performs inference based on a trained KGE model. The predictions are saved predictions.tsv in the provided
    data directory.
    :param model_directory: Directory containing the experimental artifacts: configuration.json,
    entities_to_embeddings.json, relations_to_embeddings.json and trained_model.pkl
    :param data_directory: Directory containing the candidate entities as an entities.tsv file and
    the candidate relations as relations.tsv. Both files consists of one column containint the entities/relations,
    and based on these all combinatios of possible triples are created.
    :param remove_training_triples:
    :param path_to_blacklisted_triples:
    :return:
    """
    # Load configuration file
    with open(os.path.join(model_directory, 'configuration.json')) as f:
        config = json.load(f)

    # Load entity to id mapping
    with open(os.path.join(model_directory, 'entity_to_id.json')) as f:
        entity_to_id = json.load(f)

    # Load relation to id mapping
    with open(os.path.join(model_directory, 'relation_to_id.json')) as f:
        relation_to_id = json.load(f)

    trained_kge_model: Module = get_kge_model(config=config)
    path_to_model = os.path.join(model_directory, 'trained_model.pkl')
    trained_kge_model.load_state_dict(torch.load(path_to_model))

    entities = np.loadtxt(fname=os.path.join(data_directory, 'entities.tsv'), dtype=str, delimiter='\t')
    relations = np.loadtxt(fname=os.path.join(data_directory, 'relations.tsv'), dtype=str, delimiter='\t')

    device_name = 'cuda:0' if torch.cuda.is_available() and config[PREFERRED_DEVICE] == GPU else CPU
    device = torch.device(device_name)

    ranked_triples = make_predictions(
        kge_model=trained_kge_model,
        entities=entities,
        relations=relations,
        entity_to_id=entity_to_id,
        rel_to_id=relation_to_id,
        device=device,
        blacklist_path=path_to_blacklisted_triples,
    )

    if export_predictions:
        np.savetxt(os.path.join(data_directory, 'predictions.tsv'), ranked_triples, fmt='%s')

    return ranked_triples
