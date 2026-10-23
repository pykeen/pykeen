"""Customizing the representations of a model."""

# %% customize via ERModel
from pykeen.models import ERModel
from pykeen.pipeline import pipeline

result = pipeline(
    dataset="Nations",
    model=ERModel,
    model_kwargs={
        "interaction": "TransE",
        "interaction_kwargs": {"p": 1},
        "entity_representations_kwargs": {"shape": 64, "initializer": "xavier_uniform_", "constrainer": "normalize"},
        "relation_representations_kwargs": {"shape": 64, "initializer": "xavier_uniform_norm_"},
    },
    training_kwargs={"num_epochs": 5},
)

# %% subclass for HPO
from collections.abc import Mapping
from typing import Any, ClassVar

from torch import nn

from pykeen.hpo import hpo_pipeline
from pykeen.nn import Embedding, Representation, TransformedRepresentation


class MyTransE(ERModel):
    """TransE with a choice of entity representations."""

    hpo_default: ClassVar[Mapping[str, Any]] = {
        "embedding_dim": {"type": int, "low": 4, "high": 6, "scale": "power_two"},
        "entity_representation": {"type": "categorical", "choices": ["embedding", "transformed"]},
        # only used by the transformed entity representation
        "hidden_dim": {"type": int, "low": 4, "high": 6, "scale": "power_two"},
    }

    def __init__(
        self,
        *,
        embedding_dim: int = 64,
        entity_representation: str = "embedding",
        hidden_dim: int = 32,
        **kwargs,
    ) -> None:
        """Initialize the model."""
        entity_representations: type[Representation]
        entity_representations_kwargs: dict[str, Any]
        if entity_representation == "embedding":
            entity_representations = Embedding
            entity_representations_kwargs = {"shape": embedding_dim}
        elif entity_representation == "transformed":
            entity_representations = TransformedRepresentation
            entity_representations_kwargs = {
                "transformation": nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embedding_dim),
                ),
                "base_kwargs": {"shape": embedding_dim},
            }
        else:
            raise ValueError(f"Unknown entity representation: {entity_representation}")
        super().__init__(
            interaction="TransE",
            interaction_kwargs={"p": 1},
            entity_representations=entity_representations,
            entity_representations_kwargs=entity_representations_kwargs,
            # the dimensions of entity and relation representations have to match for TransE
            relation_representations_kwargs={"shape": embedding_dim},
            **kwargs,
        )


hpo_result = hpo_pipeline(
    dataset="Nations",
    model=MyTransE,
    n_trials=4,
    training_kwargs={"num_epochs": 5},
)

# %% custom objective
import optuna

from pykeen.datasets import get_dataset

dataset = get_dataset(dataset="Nations")


def objective(trial: optuna.Trial) -> float:
    """Train and evaluate a model with a conditional search space."""
    model_kwargs: dict[str, Any] = {
        "embedding_dim": trial.suggest_categorical("embedding_dim", [16, 32, 64]),
        "entity_representation": trial.suggest_categorical("entity_representation", ["embedding", "transformed"]),
    }
    # only suggest the hidden dimension if it is used
    if model_kwargs["entity_representation"] == "transformed":
        model_kwargs["hidden_dim"] = trial.suggest_categorical("hidden_dim", [16, 32, 64])
    result = pipeline(
        # evaluate on the validation triples, and keep the test triples for the final evaluation
        training=dataset.training,
        testing=dataset.validation,
        model=MyTransE,
        model_kwargs=model_kwargs,
        training_kwargs={"num_epochs": 5},
    )
    return result.get_metric("hits@10")


# model the sub-spaces of the conditional search space separately
sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=4)
