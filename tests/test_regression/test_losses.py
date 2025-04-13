"""Regression tests for loss functions."""

import abc
import dataclasses
import json
import pathlib
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from class_resolver import Resolver

from pykeen.losses import Loss, loss_resolver

HERE = pathlib.Path(__file__).parent.resolve()
DATA_DIRECTORY = HERE.joinpath("data")
LOSSES_DIRECTORY = DATA_DIRECTORY.joinpath("losses")
LOSSES_PATH = DATA_DIRECTORY.joinpath("losses.json")


class LossTestCase(abc.ABC):
    """A test case for loss regression tests."""

    @abc.abstractmethod
    def __call__(self, instance: Loss, generator: torch.Generator) -> torch.Tensor:
        """Calculate the loss value."""
        raise NotImplementedError


@dataclasses.dataclass
class LCWATestCase(LossTestCase):
    """Test LCWA scores."""

    batch_size: int
    label_smoothing: float | None
    num_entities: int

    # docstr-coverage: inherited
    def __call__(self, instance: Loss, generator: torch.Generator) -> torch.Tensor:
        predictions = torch.rand(self.batch_size, self.num_entities, generator=generator)
        labels = (
            torch.rand(self.batch_size, self.num_entities, generator=generator).less(0.5).to(dtype=predictions.dtype)
        )
        return instance.process_lcwa_scores(
            predictions=predictions, labels=labels, label_smoothing=self.label_smoothing, num_entities=self.num_entities
        )


@dataclasses.dataclass
class SLCWATestCase(LossTestCase):
    """Test SLCWA scores."""

    batch_size: int
    label_smoothing: float | None
    num_entities: int
    num_negatives: int

    # docstr-coverage: inherited
    def __call__(self, instance: Loss, generator: torch.Generator) -> torch.Tensor:
        positive_scores = torch.rand(self.batch_size, 1, generator=generator)
        negative_scores = torch.rand(self.batch_size, self.num_negatives, generator=generator)
        return instance.process_slcwa_scores(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            label_smoothing=self.label_smoothing,
            batch_filter=None,
            num_entities=self.num_entities,
        )


test_case_resolver = Resolver(classes=[LCWATestCase, SLCWATestCase], base=LossTestCase, suffix="testcase")


# if you want to add a new configuration, you can add everything except the value,
# and it will automatically get added next time
# TODO: pytest.param is a private type...
def iter_cases(*, automatic_calculation: bool = False) -> Iterator[Any]:
    """Get loss test cases."""
    records = json.loads(LOSSES_PATH.read_text())
    should_write = False

    for record in records:
        loss = loss_resolver.make(record["loss"])

        # standardize loss key
        record["loss"] = loss_resolver.normalize_cls(loss_resolver.lookup(loss))

        # standardize type
        record["type"] = test_case_resolver.normalize_cls(test_case_resolver.lookup(record["type"]))

        loss_test_case: LossTestCase = test_case_resolver.make(record["type"], record["kwargs"])

        seed = record.get("seed")
        if seed is None:
            if not automatic_calculation:
                raise ValueError("missing seed")
            seed = record["seed"] = 42
            should_write = True

        value = record.get("value")
        if not value:
            if not automatic_calculation:
                raise ValueError("missing value")
            value = record["value"] = loss_test_case(instance=loss, generator=torch.manual_seed(seed)).item()
            should_write = True

        yield pytest.param(
            loss,
            loss_test_case,
            value,
            seed,
            id=f"{record['loss']}-{record['type']}-{record['seed']}-{record['kwargs']}",
        )

    if should_write:
        records = sorted(records, key=lambda r: (r["loss"], r["type"], r["seed"]))
        LOSSES_PATH.write_text(json.dumps(records, indent=2, sort_keys=True))


@pytest.mark.parametrize(("instance", "case", "expected", "seed"), iter_cases())
def test_regression(instance: Loss, case: LossTestCase, expected: float, seed: int) -> None:
    """Check whether the loss value is the expected one."""
    actual = case(instance=instance, generator=torch.manual_seed(seed))
    assert torch.isclose(torch.as_tensor(expected), actual)
