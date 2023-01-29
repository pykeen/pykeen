# -*- coding: utf-8 -*-

"""PyTorch Lightning integration.

PyTorch Lightning poses an alternative way to implement a training
loop and evaluation loop for knowledge graph embedding models that
has some nice features:

- mixed precision training
- multi-gpu training

.. code-block:: python

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

"""

from abc import abstractmethod
from typing import Optional

import click
import pytorch_lightning
import torch
import torch.utils.data
from class_resolver import ClassResolver, HintOrType, OptionalKwargs

from pykeen.datasets import dataset_resolver, get_dataset
from pykeen.datasets.base import Dataset
from pykeen.losses import Loss, loss_resolver
from pykeen.models import Model, model_resolver
from pykeen.models.cli import options
from pykeen.optimizers import optimizer_resolver
from pykeen.sampling import NegativeSampler
from pykeen.training import LCWATrainingLoop, SLCWATrainingLoop
from pykeen.triples.triples_factory import CoreTriplesFactory
from pykeen.typing import InductiveMode, OneOrSequence

__all__ = [
    "LitModule",
    "lit_module_resolver",
    "LCWALitModule",
    "SLCWALitModule",
]


class LitModule(pytorch_lightning.LightningModule):
    """
    A base module for training models with PyTorch Lightning.

    .. seealso::
        :class:`pykeen.training.training_loop.TrainingLoop`
    """

    def __init__(
        self,
        # dataset
        dataset: HintOrType[Dataset] = "nations",
        dataset_kwargs: OptionalKwargs = None,
        mode: Optional[InductiveMode] = None,
        # model
        model: HintOrType[Model] = "distmult",
        model_kwargs: OptionalKwargs = None,
        # stored outside of the training loop / optimizer to give access to auto-tuning from Lightning
        batch_size: int = 32,
        learning_rate: float = 1.0e-03,
        label_smoothing: float = 0.0,
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
        :param mode:
            the inductive mode; defaults to transductive training

        :param model:
            the model, or a hint thereof
        :param model_kwargs:
            additional keyword-based parameters passed to the model

        :param batch_size:
            the training batch size
        :param learning_rate:
            the learning rate
        :param label_smoothing:
            the label smoothing

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
        self.mode = mode
        self.label_smoothing = label_smoothing

    def forward(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:
        """
        Perform the prediction or inference step by wrapping :meth:`pykeen.models.ERModel.predict_t`.

        :param hr_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, relation) pairs.
        :return: shape: (batch_size, num_entities), dtype: float
            For each h-r pair, the scores for all possible tails.

        .. note::
            in lightning, forward defines the prediction/inference actions
        """
        return self.model.predict_t(hr_batch)

    @abstractmethod
    def _step(self, batch, prefix: str):
        """Perform a step and log with the given prefix."""
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        return self._step(batch, prefix="train")

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        """Perform a validation step."""
        return self._step(batch, prefix="val")

    @abstractmethod
    def _dataloader(self, triples_factory: CoreTriplesFactory, shuffle: bool = False) -> torch.utils.data.DataLoader:
        """Create a data loader."""
        raise NotImplementedError

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Create the training data loader."""
        return self._dataloader(triples_factory=self.dataset.training, shuffle=True)

    def val_dataloader(self) -> OneOrSequence[torch.utils.data.DataLoader]:
        """Create the validation data loader."""
        # TODO: In sLCWA, we still want to calculate validation *metrics* in LCWA
        if self.dataset.validation is None:
            return []
        return self._dataloader(triples_factory=self.dataset.validation, shuffle=False)

    def configure_optimizers(self):
        """Configure the optimizers."""
        return optimizer_resolver.make(
            self.optimizer, self.optimizer_kwargs, params=self.parameters(), lr=self.learning_rate
        )

    # docstr-coverage: inherited
    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:  # noqa: D102
        # call post_parameter_update
        self.model.post_parameter_update()


class SLCWALitModule(LitModule):
    """A PyTorch Lightning module for training a model with sLCWA training loop."""

    def __init__(
        self,
        *,
        negative_sampler: HintOrType[NegativeSampler] = None,
        negative_sampler_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        """
        Initialize the lightning module.

        :param negative_sampler:
            the negative sampler, cf. :meth:`pykeen.triples.CoreTriplesFactory.create_slcwa_instances`
        :param negative_sampler_kwargs:
            keyword-based parameters passed to the negative sampler, cf.
            :meth:`pykeen.triples.CoreTriplesFactory.create_slcwa_instances`
        :param kwargs:
            additional keyword-based parameters passed to :meth:`LitModule.__init__`
        """
        super().__init__(**kwargs)
        self.negative_sampler = negative_sampler
        self.negative_sampler_kwargs = negative_sampler_kwargs

    # docstr-coverage: inherited
    def _step(self, batch, prefix: str):  # noqa: D102
        loss = SLCWATrainingLoop._process_batch_static(
            model=self.model,
            loss=self.loss,
            mode=self.mode,
            batch=batch,
            label_smoothing=self.label_smoothing,
            # TODO: sub-batching / slicing
            slice_size=None,
            start=None,
            stop=None,
        )
        self.log(f"{prefix}_loss", loss)
        return loss

    # docstr-coverage: inherited
    def _dataloader(
        self, triples_factory: CoreTriplesFactory, shuffle: bool = False
    ) -> torch.utils.data.DataLoader:  # noqa: D102
        return torch.utils.data.DataLoader(
            dataset=triples_factory.create_slcwa_instances(
                batch_size=self.batch_size,
                # TODO:
                # shuffle=shuffle,
                # drop_last=drop_last,
                negative_sampler=self.negative_sampler,
                negative_sampler_kwargs=self.negative_sampler_kwargs,
                # sampler=sampler,
            ),
            # shuffle=shuffle,
            # disable automatic batching in data loader
            sampler=None,
            batch_size=None,
        )


class LCWALitModule(LitModule):
    """A PyTorch Lightning module for training a model with LCWA training loop.

    .. seealso:: https://github.com/pykeen/pykeen/pull/905
    """

    # docstr-coverage: inherited
    def _step(self, batch, prefix: str):  # noqa: D102
        loss = LCWATrainingLoop._process_batch_static(
            model=self.model,
            score_method=self.model.score_t,
            loss=self.loss,
            num_targets=self.model.num_entities,
            mode=self.mode,
            batch=batch,
            label_smoothing=self.label_smoothing,
            # TODO: sub-batching / slicing
            start=None,
            stop=None,
            slice_size=None,
        )
        self.log(f"{prefix}_loss", loss)
        return loss

    # docstr-coverage: inherited
    def _dataloader(
        self, triples_factory: CoreTriplesFactory, shuffle: bool = False
    ) -> torch.utils.data.DataLoader:  # noqa: D102
        return torch.utils.data.DataLoader(
            dataset=triples_factory.create_lcwa_instances(),
            batch_size=self.batch_size,
            shuffle=shuffle,
        )


lit_module_resolver: ClassResolver[LitModule] = ClassResolver.from_subclasses(
    base=LitModule,
    default=SLCWALitModule,
    # note: since this file is executed via __main__, its module name is replaced by __name__
    #       hence, the two classes' fully qualified names start with "_" and are considered private
    # cf. https://github.com/cthoyt/class-resolver/issues/39
    exclude_private=False,
)


def lit_pipeline(
    training_loop: HintOrType[LitModule] = None,
    training_loop_kwargs: OptionalKwargs = None,
    trainer_kwargs: OptionalKwargs = None,
) -> None:
    """
    Create a :class:`LitModule` and run :class:`pytorch_lightning.Trainer` with it.

    .. note::
        this method modifies the model's parameters in-place.

    :param training_loop:
        the training loop or a hint thereof
    :param training_loop_kwargs:
        keyword-based parameters passed to the respective :class:`LitModule` subclass upon instantiation.
    :param trainer_kwargs:
        keyword-based parameters passed to :class:`pytorch_lightning.Trainer`
    """
    pytorch_lightning.Trainer(**(trainer_kwargs or {})).fit(
        model=lit_module_resolver.make(training_loop, pos_kwargs=training_loop_kwargs)
    )


@click.command()
@lit_module_resolver.get_option("-tl", "--training-loop")
@dataset_resolver.get_option("--dataset", default="nations")
@options.inverse_triples_option
@model_resolver.get_option("-m", "--model", default="mure")
@loss_resolver.get_option("-l", "--loss", default="bcewithlogits")
@options.batch_size_option
@click.option("--embedding-dim", type=int, default=128)
@click.option("-b", "--batch-size", type=int, default=128)
@click.option("--mixed-precision", is_flag=True)
@options.number_epochs_option
def _main(
    training_loop: HintOrType[LitModule],
    dataset: HintOrType[Dataset],
    create_inverse_triples: bool,
    model: HintOrType[Model],
    loss: HintOrType[Loss],
    batch_size: int,
    embedding_dim: int,
    mixed_precision: bool,
    number_epochs: int,
):
    """Run PyTorch lightning model."""
    lit_pipeline(
        training_loop=training_loop,
        training_loop_kwargs=dict(
            dataset=dataset,
            dataset_kwargs=dict(create_inverse_triples=create_inverse_triples),
            model=model,
            model_kwargs=dict(embedding_dim=embedding_dim, loss=loss),
            batch_size=batch_size,
        ),
        trainer_kwargs=dict(
            # automatically choose accelerator
            accelerator="auto",
            # defaults to TensorBoard; explicitly disabled here
            logger=False,
            # disable checkpointing
            enable_checkpointing=False,
            # mixed precision training
            precision=16 if mixed_precision else 32,
            max_epochs=number_epochs,
        ),
    )


if __name__ == "__main__":
    _main()
