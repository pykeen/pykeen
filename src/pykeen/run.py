# -*- coding: utf-8 -*-

import json
import os
import pickle
import time
from typing import Mapping, Optional

from pykeen.constants import OUTPUT_DIREC
from pykeen.utilities.pipeline import Pipeline


def run(config: Mapping, seed: int = 2, output_directory: Optional[str] = None, training_path: Optional[str] = None):
    if output_directory is None:
        output_directory = os.path.join(config[OUTPUT_DIREC], time.strftime("%Y-%m-%d_%H:%M:%S"))

    os.makedirs(output_directory, exist_ok=True)

    # out_path = os.path.join(output_directory, 'configuration.json')
    # with open(out_path, 'w') as handle:
    #     json.dump(config, handle, indent=2)

    pipeline = Pipeline(config=config, seed=seed)

    (trained_model,
     loss_per_epoch,
     eval_summary,
     entity_to_embedding,
     relation_to_embedding,
     params) = pipeline.start(path_to_train_data=training_path)

    # TODO: Check why this doesn't work
    out_path = os.path.join(output_directory, 'configuration.json')
    with open(out_path, 'w') as handle:
        json.dump(params, handle, indent=2)

    out_path = os.path.join(output_directory, 'entities_to_embeddings.pkl')
    with open(out_path, 'wb') as handle:
        pickle.dump(entity_to_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

    out_path = os.path.join(output_directory, 'relations_to_embeddings.pkl')
    with open(out_path, 'wb') as handle:
        pickle.dump(relation_to_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

    out_path = os.path.join(output_directory, 'hyper_parameters.json')
    with open(out_path, 'w') as handle:
        for key, val in params.items():
            handle.write("%s: %s \n" % (str(key), str(val)))

    out_path = os.path.join(output_directory, 'losses.json')
    with open(out_path, 'w') as handle:
        json.dump(loss_per_epoch, handle, indent=2)

    if eval_summary is not None:
        out_path = os.path.join(output_directory, 'evaluation_summary.json')
        with open(out_path, 'w') as handle:
            json.dump(eval_summary, handle, indent=2)
