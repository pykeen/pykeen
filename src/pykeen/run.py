# -*- coding: utf-8 -*-

"""Script for starting the pipeline and saving the results."""

import json
import os
import pickle
import time
from typing import Mapping, Optional

import torch

from pykeen.constants import (
    ENTITY_TO_EMBEDDING, ENTITY_TO_ID, EVAL_SUMMARY, LOSSES, OUTPUT_DIREC, RELATION_TO_EMBEDDING, RELATION_TO_ID,
    TRAINED_MODEL,
)
from pykeen.utilities.pipeline import Pipeline


def run(config: Mapping,
        output_directory: Optional[str] = None,
        training_path: Optional[str] = None):
    """Run PyKEEN using a given configuration."""
    if output_directory is None:
        output_directory = os.path.join(config[OUTPUT_DIREC], time.strftime("%Y-%m-%d_%H:%M:%S"))
    os.makedirs(output_directory, exist_ok=True)

    pipeline = Pipeline(config=config)

    pipeline_outcome, params = pipeline.start(path_to_train_data=training_path)

    out_path = os.path.join(output_directory, 'configuration.json')
    with open(out_path, 'w') as handle:
        json.dump(params, handle, indent=2)

    out_path = os.path.join(output_directory, 'entities_to_embeddings.pkl')
    with open(out_path, 'wb') as handle:
        pickle.dump(pipeline_outcome[ENTITY_TO_EMBEDDING], handle, protocol=pickle.HIGHEST_PROTOCOL)

    out_path = os.path.join(output_directory, 'relations_to_embeddings.pkl')
    with open(out_path, 'wb') as handle:
        pickle.dump(pipeline_outcome[RELATION_TO_EMBEDDING], handle, protocol=pickle.HIGHEST_PROTOCOL)

    out_path = os.path.join(output_directory, 'entity_to_id.json')
    with open(out_path, 'w') as handle:
        json.dump(pipeline_outcome[ENTITY_TO_ID], handle, indent=2)

    out_path = os.path.join(output_directory, 'relation_to_id.json')
    with open(out_path, 'w') as handle:
        json.dump(pipeline_outcome[RELATION_TO_ID], handle, indent=2)

    out_path = os.path.join(output_directory, 'losses.json')
    with open(out_path, 'w') as handle:
        json.dump(pipeline_outcome[LOSSES], handle, indent=2)

    eval_summary = pipeline_outcome.get(EVAL_SUMMARY)
    if eval_summary is not None:
        out_path = os.path.join(output_directory, 'evaluation_summary.json')
        with open(out_path, 'w') as handle:
            json.dump(eval_summary, handle, indent=2)

    # Save trained model
    out_path = os.path.join(output_directory, 'trained_model.pkl')
    torch.save(pipeline_outcome[TRAINED_MODEL].state_dict(), out_path)
