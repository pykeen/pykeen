"""Regression tests for loss functions."""

import abc
import dataclasses
import itertools
import json
import pathlib

import pytest
import torch
from class_resolver import Resolver

from pykeen.losses import Loss, loss_resolver

HERE = pathlib.Path(__file__).parent.resolve()
DATA_DIRECTORY = HERE.joinpath("data")
LOSSES_DIRECTORY = DATA_DIRECTORY.joinpath("losses")
LOSSES_PATH = DATA_DIRECTORY.joinpath("losses.json")


@pytest.fixture
def generator() -> torch.Generator:
    """Create a generator with fixed seed."""
    return torch.manual_seed(42)


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


def get_cases() -> list[tuple[Loss, LossTestCase, float, int]]:
    """Get loss test cases."""
    rv = []
    for data in json.loads(LOSSES_PATH.read_text()):
        loss = loss_resolver.make(data["loss"])
        test_case: LossTestCase = test_case_resolver.make(data["type"], data["kwargs"])
        value = data["value"]
        seed = data["seed"]
        rv.append((loss, test_case, value, seed))
    return rv


def _ids(x: type | LossTestCase) -> str:
    """Determine part of test case name."""
    if isinstance(x, LossTestCase):
        return str(x)
    return x.__name__


@pytest.mark.parametrize(("instance", "case", "expected", "seed"), get_cases())
def test_regression_2(instance: Loss, case: LossTestCase, expected: float, seed: int) -> None:
    """Check whether the loss value is the expected one."""
    actual = case(instance=instance, generator=torch.manual_seed(seed))
    assert torch.isclose(torch.as_tensor(expected), actual)


@pytest.mark.parametrize(
    ("cls", "case"),
    itertools.product(
        set(loss_resolver),
        [
            LCWATestCase(batch_size=1, label_smoothing=None, num_entities=32),
            SLCWATestCase(batch_size=1, label_smoothing=None, num_entities=32, num_negatives=3),
        ],
    ),
    ids=_ids,
)
def test_regression(cls: type[Loss], case: LossTestCase, generator: torch.Generator) -> None:
    """Check whether the loss value is the expected one."""
    # create instance with default parameters.
    instance = loss_resolver.make(cls)

    # get loss value for the given case
    loss_value = case(instance=instance, generator=generator)

    # determine reference file path
    name = loss_resolver.normalize_cls(cls)
    path = LOSSES_DIRECTORY.joinpath(name).with_suffix(".json")

    # TODO: is there a nicer way how to enable generating missing values more explicitly?
    # load expected value
    references = {}
    if path.is_file():
        with path.open() as file:
            references = json.load(file)
    key = str(case)
    if key not in references:
        references[str(case)] = loss_value.item()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode="w") as file:
            json.dump(references, file, indent=2, sort_keys=True)

    # compare for approximate equivalence
    reference_value = torch.as_tensor(references[str(case)])
    assert torch.isclose(loss_value, reference_value)
