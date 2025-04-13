"""Regression tests for loss functions."""

import abc
import dataclasses
import json
import logging
import pathlib
from collections.abc import Iterable, Iterator
from typing import Any, TypedDict

import click
import pytest
import torch
from class_resolver import ClassResolver

from pykeen.losses import Loss, loss_resolver

HERE = pathlib.Path(__file__).parent.resolve()
DATA_DIRECTORY = HERE.joinpath("data")
LOSSES_PATH = DATA_DIRECTORY.joinpath("losses.json")

logger = logging.getLogger(__name__)


class LossTestCase(abc.ABC):
    """A test case for loss regression tests."""

    @abc.abstractmethod
    def __call__(self, instance: Loss, generator: torch.Generator) -> torch.Tensor:
        """Calculate the loss value."""
        raise NotImplementedError


@dataclasses.dataclass
class LCWALossTestCase(LossTestCase):
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
class SLCWALossTestCase(LossTestCase):
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


test_case_resolver: ClassResolver[LossTestCase] = ClassResolver.from_subclasses(base=LossTestCase)


class Record(TypedDict):
    """A regression test case record."""

    type: str
    kwargs: dict[str, Any]
    seed: int

    values: dict[str, float]


def iter_records(path: pathlib.Path) -> Iterator[Record]:
    """Iterate records from path."""
    with path.open() as file:
        yield from json.load(file)


def save_records(path: pathlib.Path, records: Iterable[Record]) -> None:
    """Save records to path."""
    records = sorted(records, key=lambda r: (r["type"], r["seed"], json.dumps(r["kwargs"])))
    with path.open(mode="w") as file:
        json.dump(records, file, indent=2, sort_keys=True)


# TODO: pytest.param is a private type...
def iter_cases() -> Iterator[Any]:
    """Get loss test cases."""
    for record in iter_records(LOSSES_PATH):
        loss_test_case = test_case_resolver.make(record["type"], record["kwargs"])
        for name, value in record["values"].items():
            loss = loss_resolver.make(name)
            yield pytest.param(
                loss,
                loss_test_case,
                value,
                record["seed"],
                id=f"{name}-{record['type']}-{record['seed']}-{record['kwargs']}",
            )


@pytest.mark.parametrize(("instance", "case", "expected", "seed"), iter_cases())
def test_regression(instance: Loss, case: LossTestCase, expected: float, seed: int) -> None:
    """Check whether the loss value is the expected one."""
    actual = case(instance=instance, generator=torch.manual_seed(seed))
    assert torch.isclose(torch.as_tensor(expected), actual)


@click.command()
@click.option("--path", type=pathlib.Path, default=LOSSES_PATH)
def update(path: pathlib.Path) -> None:
    """Write test cases for all losses."""
    logging.basicConfig(level=logging.INFO)

    # determine unique settings (using JSON-representation)
    unique_cases_jsons: set[str] = set()
    total = 0
    keys = {"seed", "type", "kwargs"}
    for record in iter_records(path):
        total += 1
        unique_cases_jsons.add(json.dumps({key: record[key] for key in keys}))
    logger.info(f"Found {len(unique_cases_jsons):_} unique setings at {path!s}")

    # create case for full cartesian product between cases & losses
    records: list[Record] = list()
    for unique_case_json in unique_cases_jsons:
        data = json.loads(unique_case_json)
        data["values"] = values = {}
        case = test_case_resolver.make(data["type"], data["kwargs"])
        for cls in loss_resolver:
            instance = loss_resolver.make(cls)
            key = loss_resolver.normalize_cls(cls)
            value = case(instance=instance, generator=torch.manual_seed(data["seed"]))
            values[key] = float(value)
        records.append(data)
    save_records(path, records)
    logger.info(f"Written {len(records):_} records to {path!s}")


if __name__ == "__main__":
    update()
