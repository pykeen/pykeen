# -*- coding: utf-8 -*-

'''Script for starting the pipeline and saving the results.'''

import json
import os
import pickle
import time
from typing import Mapping, Optional

import torch

from pykeen.constants import OUTPUT_DIREC, ENTITY_TO_EMBEDDING, RELATION_TO_EMBEDDING, LOSSES, EVAL_SUMMARY, \
    TRAINED_MODEL, ENTITY_TO_ID, RELATION_TO_ID
from pykeen.utilities.pipeline import Pipeline


def run(config: Mapping, seed: int = 2, output_directory: Optional[str] = None, training_path: Optional[str] = None):
    if output_directory is None:
        output_directory = os.path.join(config[OUTPUT_DIREC], time.strftime("%Y-%m-%d_%H:%M:%S"))

    os.makedirs(output_directory, exist_ok=True)

    pipeline = Pipeline(config=config, seed=seed)

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

    out_path = os.path.join(output_directory, 'hyper_parameters.json')
    with open(out_path, 'w') as handle:
        for key, val in params.items():
            handle.write("%s: %s \n" % (str(key), str(val)))

    out_path = os.path.join(output_directory, 'losses.json')
    with open(out_path, 'w') as handle:
        json.dump(pipeline_outcome[LOSSES], handle, indent=2)

    eval_summary = pipeline_outcome[EVAL_SUMMARY]
    if eval_summary is not None:
        out_path = os.path.join(output_directory, 'evaluation_summary.json')
        with open(out_path, 'w') as handle:
            json.dump(eval_summary, handle, indent=2)

    # Save trained model
    out_path = os.path.join(output_directory, 'trained_model.pkl')
    torch.save(pipeline_outcome[TRAINED_MODEL].state_dict(), out_path)
