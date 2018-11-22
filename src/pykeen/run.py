# -*- coding: utf-8 -*-

"""Script for starting the pipeline and saving the results."""

import json
import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch

from pykeen.constants import (
    ENTITY_TO_EMBEDDING, ENTITY_TO_ID, EVAL_SUMMARY, LOSSES, OUTPUT_DIREC, RELATION_TO_EMBEDDING, RELATION_TO_ID,
    TRAINED_MODEL,
)
from pykeen.utilities.pipeline import Pipeline


@dataclass
class Results:
    """Results from PyKEEN."""

    config: Mapping
    pipeline: Pipeline
    results: Mapping


def run(config: Dict,
        output_directory: Optional[str] = None) -> Results:
    """Run PyKEEN using a given configuration."""
    if output_directory is None:
        output_directory = os.path.join(config[OUTPUT_DIREC], time.strftime("%Y-%m-%d_%H:%M:%S"))
    os.makedirs(output_directory, exist_ok=True)

    pipeline = Pipeline(config=config)
    pipeline_results = pipeline.run()

    with open(os.path.join(output_directory, 'configuration.json'), 'w') as file:
        json.dump(pipeline.config, file, indent=2)

    with open(os.path.join(output_directory, 'entities_to_embeddings.pkl'), 'wb') as file:
        pickle.dump(pipeline_results[ENTITY_TO_EMBEDDING], file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_directory, 'entities_to_embeddings.json'), 'w') as file:
        json.dump(
            {
                key: list(map(float, array))
                for key, array in pipeline_results[ENTITY_TO_EMBEDDING].items()
            },
            file,
            indent=2,
            sort_keys=True
        )

    with open(os.path.join(output_directory, 'relations_to_embeddings.pkl'), 'wb') as file:
        pickle.dump(pipeline_results[RELATION_TO_EMBEDDING], file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_directory, 'relations_to_embeddings.json'), 'w') as file:
        json.dump(
            {
                key: list(map(float, array))
                for key, array in pipeline_results[RELATION_TO_EMBEDDING].items()
            },
            file,
            indent=2,
            sort_keys=True,
        )

    with open(os.path.join(output_directory, 'entity_to_id.json'), 'w') as file:
        json.dump(pipeline_results[ENTITY_TO_ID], file, indent=2, sort_keys=True)

    with open(os.path.join(output_directory, 'relation_to_id.json'), 'w') as file:
        json.dump(pipeline_results[RELATION_TO_ID], file, indent=2, sort_keys=True)

    with open(os.path.join(output_directory, 'losses.json'), 'w') as file:
        json.dump(pipeline_results[LOSSES], file, indent=2, sort_keys=True)

    eval_summary = pipeline_results.get(EVAL_SUMMARY)
    if eval_summary is not None:
        with open(os.path.join(output_directory, 'evaluation_summary.json'), 'w') as file:
            json.dump(eval_summary, file, indent=2)

    # Save trained model
    torch.save(
        pipeline_results[TRAINED_MODEL].state_dict(),
        os.path.join(output_directory, 'trained_model.pkl'),
    )

    return Results(
        config=config,
        pipeline=pipeline,
        results=pipeline_results,
    )
