"""Regression tests for loss functions."""

import abc
import dataclasses
import json
import logging
import pathlib
from collections.abc import Iterable, Iterator
from typing import Any, NamedTuple, TypedDict

import click
import pytest
import torch
from class_resolver import ClassResolver

from pykeen.losses import Loss, NoSampleWeightSupportError, UnsupportedLabelSmoothingError, loss_resolver

HERE = pathlib.Path(__file__).parent.resolve()
DATA_DIRECTORY = HERE.joinpath("data")
LOSSES_PATH = DATA_DIRECTORY.joinpath("losses.json")

logger = logging.getLogger(__name__)


class LossCalculator(abc.ABC):
    """Calculate loss values on randomized input for regression tests."""

    @abc.abstractmethod
    def __call__(self, instance: Loss, generator: torch.Generator) -> torch.Tensor:
        """Calculate the loss value."""
        raise NotImplementedError


@dataclasses.dataclass
class LCWALossCalculator(LossCalculator):
    """Calculate loss values for LCWA."""

    batch_size: int

    label_smoothing: float | None
    num_entities: int

    weighted: bool = False

    # docstr-coverage: inherited
    def __call__(self, instance: Loss, generator: torch.Generator) -> torch.Tensor:
        predictions = torch.rand(self.batch_size, self.num_entities, generator=generator)
        labels = (
            torch.rand(self.batch_size, self.num_entities, generator=generator).less(0.5).to(dtype=predictions.dtype)
        )
        if self.weighted:
            weights = torch.rand(self.batch_size, self.num_entities, generator=generator)
        else:
            weights = None
        return instance.process_lcwa_scores(
            predictions=predictions,
            labels=labels,
            label_smoothing=self.label_smoothing,
            num_entities=self.num_entities,
            weights=weights,
        )


@dataclasses.dataclass
class SLCWALossCalculator(LossCalculator):
    """Calculate loss values for sLCWA."""

    batch_size: int

    label_smoothing: float | None
    num_entities: int

    num_negatives: int

    weighted: bool = False

    # docstr-coverage: inherited
    def __call__(self, instance: Loss, generator: torch.Generator) -> torch.Tensor:
        positive_scores = torch.rand(self.batch_size, 1, generator=generator)
        negative_scores = torch.rand(self.batch_size, self.num_negatives, generator=generator)
        if self.weighted:
            pos_weights = torch.rand(self.batch_size, 1, generator=generator)
            neg_weights = torch.rand(self.batch_size, self.num_negatives, generator=generator)
        else:
            pos_weights = None
            neg_weights = None
        return instance.process_slcwa_scores(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            label_smoothing=self.label_smoothing,
            batch_filter=None,
            num_entities=self.num_entities,
            pos_weights=pos_weights,
            neg_weights=neg_weights,
        )


calculator_resolver: ClassResolver[LossCalculator] = ClassResolver(
    [SLCWALossCalculator, LCWALossCalculator], base=LossCalculator
)


class Record(TypedDict):
    """A regression test case record."""

    #: the class name of the calculator
    type: str
    #: keyword parameters for the calculator
    kwargs: dict[str, Any]

    #: the seed for randomized input
    seed: int
    #: a mapping from loss class name to value
    loss_name_to_value: dict[str, float]


def iter_records(path: pathlib.Path) -> Iterator[Record]:
    """Iterate records from path."""
    with path.open() as file:
        yield from json.load(file)


def save_records(path: pathlib.Path, records: Iterable[Record]) -> None:
    """Save records to path."""
    records = sorted(records, key=lambda r: (r["type"], r["seed"], json.dumps(r["kwargs"], sort_keys=True)))
    with path.open(mode="w") as file:
        json.dump(records, file, indent=2, sort_keys=True)


class Case(NamedTuple):
    """An individual test case."""

    calculator: LossCalculator
    loss: Loss
    loss_name: str
    seed: int
    expected: float


def iter_cases(record: Record) -> Iterator[Case]:
    """Iterate over individual test cases."""
    calculator = calculator_resolver.make(record["type"], record["kwargs"])
    for name, value in record.get("loss_name_to_value", {}).items():
        loss = loss_resolver.make(name)
        yield Case(calculator=calculator, loss=loss, loss_name=name, seed=record["seed"], expected=value)


@pytest.mark.parametrize(
    ("instance", "case", "expected", "seed"),
    [
        pytest.param(
            case.loss,
            case.calculator,
            case.expected,
            case.seed,
            id=f"{case.loss_name}-{record['type']}-{record['kwargs']}",
        )
        for record in iter_records(LOSSES_PATH)
        for case in iter_cases(record)
    ],
)
def test_regression(instance: Loss, case: LossCalculator, expected: float, seed: int) -> None:
    """Check whether the loss value is the expected one."""
    actual = case(instance=instance, generator=torch.manual_seed(seed))
    assert torch.isclose(torch.as_tensor(expected), actual, atol=1e-5)


@click.command()
@click.option("--path", type=pathlib.Path, default=LOSSES_PATH)
@click.option("--digits", type=int, default=6)
def update(path: pathlib.Path, digits: int) -> None:
    """Write test cases for all losses."""
    logging.basicConfig(level=logging.INFO)

    # determine unique settings (using JSON-representation)
    unique_cases_jsons: set[str] = set()
    total = 0
    keys = {"seed", "type", "kwargs"}
    for record in iter_records(path):
        total += 1
        unique_cases_jsons.add(json.dumps({key: record[key] for key in keys}, sort_keys=True))
    logger.info(f"Found {len(unique_cases_jsons):_} unique settings at {path!s}")

    # create case for full cartesian product between cases & losses
    records: list[Record] = list()
    for unique_case_json in unique_cases_jsons:
        data = json.loads(unique_case_json)
        data["loss_name_to_value"] = loss_name_to_value = {}
        case = calculator_resolver.make(data["type"], data["kwargs"])
        for cls in loss_resolver:
            instance = loss_resolver.make(cls)
            key = loss_resolver.normalize_cls(cls)
            try:
                value = case(instance=instance, generator=torch.manual_seed(data["seed"]))
            except (UnsupportedLabelSmoothingError, NoSampleWeightSupportError):
                continue
            loss_name_to_value[key] = round(float(value), digits)
        records.append(data)
    save_records(path, records)
    logger.info(f"Written {len(records):_} records to {path!s}")


if __name__ == "__main__":
    update()
