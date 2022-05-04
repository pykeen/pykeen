"""PyTorch Lightning integration."""
import pytorch_lightning
import torch
import torch.utils.data
from class_resolver import HintOrType, OptionalKwargs

from pykeen.datasets import get_dataset
from pykeen.datasets.base import Dataset
from pykeen.models import Model, model_resolver
from pykeen.optimizers import optimizer_resolver
from pykeen.triples.triples_factory import CoreTriplesFactory


class LitLCWAModule(pytorch_lightning.LightningModule):
    """A PyTorch Lightning module for training a model with LCWA training loop."""

    def __init__(
        self,
        # dataset
        dataset: HintOrType[Dataset] = "nations",
        dataset_kwargs: OptionalKwargs = None,
        # model
        model: HintOrType[Model] = "distmult",
        model_kwargs: OptionalKwargs = None,
        # stored outside of the training loop / optimizer to give access to auto-tuning from Lightning
        batch_size: int = 32,
        learning_rate: float = 1.0e-03,
        # optimizer
        optimizer: HintOrType[torch.optim.Optimizer] = None,
        optimizer_kwargs: OptionalKwargs = None,
    ):
        """
        Create the lightning module.

        :param dataset:
            the dataset, or a hint thereof
        :param dataset_kwargs:
            additional keyword-based parameters passed to the dataset

        :param model:
            the model, or a hint thereof
        :param model_kwargs:
            additional keyword-based parameters passed to the model

        :param batch_size:
            the training batch size
        :param learning_rate:
            the learning rate

        :param optimizer:
            the optimizer, or a hint thereof
        :param optimizer_kwargs:
            additional keyword-based parameters passed to the optimizer. should not contain `lr`, or `params`.
        """
        super().__init__()
        self.dataset = get_dataset(dataset=dataset, dataset_kwargs=dataset_kwargs)
        self.model = model_resolver.make(model, model_kwargs, triples_factory=self.dataset.training)
        self.loss = self.model.loss
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def forward(self, x):
        """
        Perform the prediction or inference step.

        .. note::
            in lightning, forward defines the prediction/inference actions
        """
        return self.model.predict_t(x)

    def _step(self, batch, prefix: str):
        """Refactored step."""
        hr_batch, labels = batch
        scores = self.model.score_t(hr_batch=hr_batch)
        loss = self.loss.process_lcwa_scores(predictions=scores, labels=labels)
        self.log(f"{prefix}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        return self._step(batch, prefix="train")

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        """Perform a validation step."""
        return self._step(batch, prefix="val")

    def _dataloader(self, triples_factory: CoreTriplesFactory, shuffle: bool = False) -> torch.utils.data.DataLoader:
        """Create a data loader."""
        return torch.utils.data.DataLoader(
            dataset=triples_factory.create_lcwa_instances(),
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        """Create the training data loader."""
        return self._dataloader(triples_factory=self.dataset.training, shuffle=True)

    def val_dataloader(self):
        """Create the validation data loader."""
        return self._dataloader(triples_factory=self.dataset.validation, shuffle=False)

    def configure_optimizers(self):
        """Configure the optimizers."""
        return optimizer_resolver.make(
            self.optimizer, self.optimizer_kwargs, params=self.parameters(), lr=self.learning_rate
        )


def main():
    """Run PyTorch lightning model."""
    model = LitLCWAModule(
        dataset="fb15k237",
        dataset_kwargs=dict(create_inverse_triples=True),
        model="mure",
        model_kwargs=dict(embedding_dim=128, loss="bcewithlogits"),
        batch_size=128,
    )
    trainer = pytorch_lightning.Trainer(
        accelerator="auto",  # automatically choose accelerator
        logger=False,  # defaults to TensorBoard; explicitly disabled here
        precision=16,  # mixed precision training
    )
    trainer.fit(model=model)


if __name__ == "__main__":
    main()
