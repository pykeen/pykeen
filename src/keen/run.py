# -*- coding: utf-8 -*-

import json
import os
import pickle
import time
from typing import Mapping

from keen.constants import OUTPUT_DIREC
from keen.utilities.pipeline import Pipeline


def run(config: Mapping, seed: int = 2):
    current_time = time.strftime("%H:%M:%S")
    current_date = time.strftime("%d/%m/%Y").replace('/', '-')
    output_direc = config[OUTPUT_DIREC]
    output_direc = os.path.join(output_direc, current_date + '_' + current_time + '')

    os.makedirs(output_direc, exist_ok=True)

    out_path = os.path.join(output_direc, 'configuration.json')
    with open(out_path, 'w') as handle:
        json.dump(config, handle, indent=2)

    pipeline = Pipeline(config=config, seed=seed)

    trained_model, loss_per_epoch, eval_summary, entity_to_embedding, relation_to_embedding, params = pipeline.start()

    out_path = os.path.join(output_direc, 'entities_to_embeddings.pkl')
    with open(out_path, 'wb') as handle:
        pickle.dump(entity_to_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

    out_path = os.path.join(output_direc, 'relations_to_embeddings.pkl')
    with open(out_path, 'wb') as handle:
        pickle.dump(relation_to_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)


    out_path = os.path.join(output_direc, 'hyper_parameters.json')
    with open(out_path, 'w') as handle:
        for key, val in params.items():
            handle.write("%s: %s \n" % (str(key), str(val)))

    out_path = os.path.join(output_direc, 'losses.json')
    with open(out_path, 'w') as handle:
        json.dump(loss_per_epoch, handle, indent=2)

    if eval_summary != None:
        out_path = os.path.join(output_direc, 'evaluation_summary.json')
        with open(out_path, 'w') as handle:
            json.dump(eval_summary, handle, indent=2)
