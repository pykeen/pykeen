# -*- coding: utf-8 -*-

"""Utilities for reproducing experiments."""

import json
import logging
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

import pandas as pd

from ..pipeline import PipelineResult, pipeline

__all__ = [
    'pipeline_from_path',
    'PipelineResultSet',
]

logger = logging.getLogger(__name__)


def pipeline_from_path(
    path: str,
    mlflow_tracking_uri: Optional[str] = None,
) -> PipelineResult:
    """Run the pipeline with configuration in a JSON file at the given path.

    :param path: The path to an experiment JSON file
    :param mlflow_tracking_uri: The URL of the MLFlow tracking server. If None, do not use MLFlow for result tracking.
    """
    with open(path) as file:
        config = json.load(file)

    metadata, pipeline_kwargs = config['metadata'], config['pipeline']
    title = metadata.get('title')
    if title is not None:
        logger.info(f'Running: {title}')

    return pipeline(
        mlflow_tracking_uri=mlflow_tracking_uri,
        metadata=metadata,
        **pipeline_kwargs,
    )


@dataclass
class PipelineResultSet:
    """A set of results."""

    pipeline_results: List[PipelineResult]

    @classmethod
    def from_path(cls, path: str, replicates: int = 10) -> 'PipelineResultSet':
        """Run the same pipeline several times."""
        return cls([
            pipeline_from_path(path)
            for _ in range(replicates)
        ])

    def get_loss_df(self) -> pd.DataFrame:
        """Get the losses as a dataframe."""
        return pd.DataFrame(
            [
                (replicate, epoch, loss)
                for replicate, result in enumerate(self.pipeline_results, start=1)
                for epoch, loss in enumerate(result.losses, start=1)
            ],
            columns=['Replicate', 'Epoch', 'Loss'],
        )

    def plot_losses(self, sns_kwargs: Optional[Mapping[str, Any]] = None):
        """Plot the several losses."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = self.get_loss_df()
        sns.set()
        if self.pipeline_results[0].title is not None:
            plt.title(self.pipeline_results[0].title)
        return sns.lineplot(data=df, x='Epoch', y='Loss', **(sns_kwargs or {}))
