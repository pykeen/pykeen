# -*- coding: utf-8 -*-

"""Script for starting the pipeline and saving the results."""

import json
import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import torch

from pykeen.constants import (
    ENTITY_TO_EMBEDDING, ENTITY_TO_ID, EVAL_SUMMARY, FINAL_CONFIGURATION, LOSSES, OUTPUT_DIREC, RELATION_TO_EMBEDDING,
    RELATION_TO_ID, TRAINED_MODEL, VERSION,
)
from pykeen.utilities.pipeline import Pipeline

__all__ = [
    'Results',
    'run',
]


@dataclass
class Results:
    """Results from PyKEEN."""

    #: The configuration used to train the KGE model
    config: Mapping

    #: The pipeline used to train the KGE model
    pipeline: Pipeline

    #: The results of training the KGE model
    results: Mapping

    @property
    def trained_model(self) -> torch.nn.Module:  # noqa: D401
        """The pre-trained KGE model."""
        return self.results['trained_model']

    @property
    def losses(self):  # noqa: D401
        """The losses calculated during training."""
        return self.results['losses']

    @property
    def evaluation_summary(self):  # noqa: D401
        """The evaluation summary."""
        return self.results['eval_summary']

    def plot_losses(self) -> None:
        """Plot the losses using Matplotlib."""
        import matplotlib.pyplot as plt
        epochs = np.arange(len(self.losses))
        plt.title('Loss Per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(epochs, self.losses)


def export_experimental_artifacts(
        results: Mapping,
        output_directory: str,
) -> None:
    """Export export experimental artifacts."""

    with open(os.path.join(output_directory, 'configuration.json'), 'w') as file:
        # In HPO model initial configuration is different from final configurations, that's why we differentiate
        json.dump(results[FINAL_CONFIGURATION], file, indent=2)

    with open(os.path.join(output_directory, 'entities_to_embeddings.pkl'), 'wb') as file:
        pickle.dump(results[ENTITY_TO_EMBEDDING], file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_directory, 'entities_to_embeddings.json'), 'w') as file:
        json.dump(
            {
                key: list(map(float, array))
                for key, array in results[ENTITY_TO_EMBEDDING].items()
            },
            file,
            indent=2,
            sort_keys=True,
        )

    if results[RELATION_TO_EMBEDDING] is not None:
        with open(os.path.join(output_directory, 'relations_to_embeddings.pkl'), 'wb') as file:
            pickle.dump(results[RELATION_TO_EMBEDDING], file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(output_directory, 'relations_to_embeddings.json'), 'w') as file:
            json.dump(
                {
                    key: list(map(float, array))
                    for key, array in results[RELATION_TO_EMBEDDING].items()
                },
                file,
                indent=2,
                sort_keys=True,
            )

    with open(os.path.join(output_directory, 'entity_to_id.json'), 'w') as file:
        json.dump(results[ENTITY_TO_ID], file, indent=2, sort_keys=True)

    with open(os.path.join(output_directory, 'relation_to_id.json'), 'w') as file:
        json.dump(results[RELATION_TO_ID], file, indent=2, sort_keys=True)

    with open(os.path.join(output_directory, 'losses.json'), 'w') as file:
        json.dump(results[LOSSES], file, indent=2, sort_keys=True)

    eval_summary = results.get(EVAL_SUMMARY)
    if eval_summary is not None:
        with open(os.path.join(output_directory, 'evaluation_summary.json'), 'w') as file:
            json.dump(eval_summary, file, indent=2)

    # Save trained model
    torch.save(
        results[TRAINED_MODEL].state_dict(),
        os.path.join(output_directory, 'trained_model.pkl'),
    )


def run(
        config: Dict,
        output_directory: Optional[str] = None,
) -> Results:
    """Train a KGE model.

    :param config: The configuration specifying the KGE model and its hyper-parameters
    :param output_directory: The directory to store the results
    """

    if output_directory is None:
        if OUTPUT_DIREC not in config:
            raise Exception('No output directory defined.')
        output_directory = os.path.join(config[OUTPUT_DIREC], time.strftime("%Y-%m-%d-%H-%M-%S"))

    os.makedirs(output_directory, exist_ok=True)

    config['pykeen-version'] = VERSION

    pipeline = Pipeline(config=config)
    results = pipeline.run()

    # Export experimental artifacts
    export_experimental_artifacts(
        results=results,
        output_directory=output_directory,
    )

    return Results(
        config=config,
        pipeline=pipeline,
        results=results,
    )
