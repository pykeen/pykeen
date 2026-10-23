"""Generic tests which inspect the package as a whole instead of testing a single component.

The checks in here are deliberately independent of concrete classes: they iterate over all modules, metrics or
classes, and assert a package-wide invariant. Cases which are known to violate an invariant are marked as strict
``xfail``; whoever fixes the underlying problem has to remove the corresponding mark, since a fixed case would
otherwise fail as an unexpected pass.
"""

import abc
import importlib
import inspect
import json
import pathlib
import subprocess
import sys
from collections.abc import Collection, Iterable, Mapping
from typing import Any

import pytest
import torch

import pykeen
from pykeen.evaluation.classification_evaluator import ClassificationMetricKey, ClassificationMetricResults
from pykeen.evaluation.rank_based_evaluator import RankBasedMetricKey, RankBasedMetricResults
from pykeen.metrics.classification import classification_metric_resolver
from pykeen.metrics.ranking import HITS_METRICS, RankBasedMetric, rank_based_metric_resolver
from pykeen.nn.combination import Combination
from pykeen.nn.message_passing import Decomposition
from pykeen.nn.representation import Embedding, Representation
from pykeen.typing import RANK_TYPES, SIDES
from pykeen.utils import ExtraReprMixin

PACKAGE_ROOT = pathlib.Path(pykeen.__file__).parent

# modules which must not be imported just for the sake of checking them
SKIPPED_MODULES = {"pykeen.__main__"}


def iter_module_names() -> Iterable[str]:
    """Iterate over the names of all modules and packages in pykeen, without importing them."""
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        parts = list(path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        name = ".".join(parts)
        if name not in SKIPPED_MODULES:
            yield name


MODULE_NAMES = sorted(iter_module_names())

# Importing these modules in a fresh interpreter fails, since the `pykeen.sampling` package and
# `pykeen.triples.instances` import each other, and the former fails if it is imported first.
CIRCULAR_IMPORT_MODULES = {
    "pykeen.sampling",
    "pykeen.sampling.basic_negative_sampler",
    "pykeen.sampling.bernoulli_negative_sampler",
    "pykeen.sampling.filtering",
    "pykeen.sampling.negative_sampler",
    "pykeen.sampling.pseudo_type",
}

# The worker imports the third-party dependencies once, and then, for each module, drops all `pykeen` modules from
# `sys.modules` before importing the module. This gives the same result as one interpreter per module, but needs
# seconds instead of minutes, since the (slow) import of the third-party dependencies is shared.
_IMPORT_WORKER = """
import importlib, json, sys

names = json.loads(sys.argv[1])
import pykeen.pipeline  # noqa: F401  (warm up the third-party imports)

results = {}
for name in names:
    for key in [key for key in sys.modules if key == "pykeen" or key.startswith("pykeen.")]:
        del sys.modules[key]
    try:
        importlib.import_module(name)
    except BaseException as error:
        results[name] = {"type": type(error).__name__, "name": getattr(error, "name", None), "message": str(error)}
print("RESULTS=" + json.dumps(results))
"""


@pytest.fixture(scope="module")
def fresh_import_results(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, Mapping[str, Any]]:
    """Import each module in an otherwise fresh set of pykeen modules; return the errors by module name."""
    process = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _IMPORT_WORKER, json.dumps(MODULE_NAMES)],
        capture_output=True,
        text=True,
        check=False,
        # avoid picking up the current working directory (i.e., the repository root) in `sys.path`
        cwd=tmp_path_factory.mktemp("fresh_import"),
    )
    lines = [line for line in process.stdout.splitlines() if line.startswith("RESULTS=")]
    assert lines, f"The import worker crashed:\n{process.stderr}"
    return json.loads(lines[-1].removeprefix("RESULTS="))


@pytest.mark.parametrize(
    "module_name",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                strict=True,
                reason="circular import between pykeen.sampling and pykeen.triples.instances",
            ),
        )
        if name in CIRCULAR_IMPORT_MODULES
        else name
        for name in MODULE_NAMES
    ],
)
def test_module_imports_in_fresh_interpreter(module_name: str, fresh_import_results: Mapping[str, Mapping[str, Any]]):
    """Test that every module can be imported first, i.e., without any other pykeen module imported before."""
    error = fresh_import_results.get(module_name)
    if error is None:
        return
    # optional third-party dependency which is not installed
    if error["type"] == "ModuleNotFoundError" and not (error["name"] or "").startswith("pykeen"):
        pytest.skip(f"Missing optional dependency: {error['name']}")
    pytest.fail(f"Importing {module_name} failed with {error['type']}: {error['message']}")


# --- metric keys ---------------------------------------------------------------------------------------------------


def test_default_rank_based_metric_keys_are_unique():
    """Test that different rank-based metrics (with default parameters) have different keys."""
    keys = [metric_cls().key for metric_cls in set(rank_based_metric_resolver.lookup_dict.values())]
    assert len(keys) == len(set(keys)), sorted(key for key in keys if keys.count(key) > 1)


def test_default_classification_metric_keys_are_unique():
    """Test that different classification metrics have different keys."""
    keys = [metric_cls().key for metric_cls in set(classification_metric_resolver.lookup_dict.values())]
    assert len(keys) == len(set(keys)), sorted(key for key in keys if keys.count(key) > 1)


def _rank_based_metric_names() -> list[str]:
    """Get the (key of the) metrics which are contained in randomly generated rank-based results."""
    return sorted({key.metric for key in RankBasedMetricResults.create_random(random_state=0).data})


def _supports_all_rank_types(metric_name: str) -> bool:
    """Return whether the rank-based metric with the given (default-parameter) key supports all rank types."""
    for metric_cls in rank_based_metric_resolver.lookup_dict.values():
        metric = metric_cls()
        if metric.key == metric_name:
            return set(metric.supported_rank_types) == set(RANK_TYPES)
    raise AssertionError(f"Unknown metric {metric_name}")


@pytest.mark.parametrize(
    "metric_name",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                strict=True,
                reason="results contain rank types which the metric does not support, and cannot be read back",
            ),
        )
        if not _supports_all_rank_types(name)
        else name
        for name in _rank_based_metric_names()
    ],
)
def test_rank_based_result_keys_round_trip(metric_name: str):
    """Test that every key in the results can be converted to a string and back, and be used to access the value."""
    results = RankBasedMetricResults.create_random(random_state=0)
    keys = [key for key in results.data if key.metric == metric_name]
    assert keys
    for key in keys:
        string = RankBasedMetricResults.key_to_string(key)
        assert RankBasedMetricResults.key_from_string(string) == key
        assert results.get_metric(string) == results.get_metric(key)


@pytest.mark.parametrize("metric_cls", sorted(classification_metric_resolver.lookup_dict.values(), key=str))
def test_classification_result_keys_round_trip(metric_cls: type):
    """Test that classification metric keys can be converted to a string and back."""
    for side in sorted(SIDES):
        key = ClassificationMetricKey(side=side, metric=metric_cls().key)
        assert ClassificationMetricResults.key_from_string(ClassificationMetricResults.key_to_string(key)) == key


# Synonyms which contain a placeholder for k are not parsed correctly if an explicit k follows.
UNPARSED_SYNONYMS_WITH_K = {
    ("AdjustedHitsAtK", "ah@5"),
    ("AdjustedHitsAtK", "ahits@5"),
    ("AdjustedHitsAtK", "ahits_at_5"),
    ("AdjustedHitsAtK", "ah_at_5"),
    ("HitsAtK", "h_at_5"),
}


def _iter_rank_based_synonym_cases() -> Iterable[tuple[type[RankBasedMetric], str, str]]:
    """Iterate over (metric class, query, expected key) for all synonyms of rank-based metrics."""
    for metric_cls in sorted(set(rank_based_metric_resolver.lookup_dict.values()), key=lambda cls: cls.__name__):
        for synonym in metric_cls.synonyms:
            if issubclass(metric_cls, HITS_METRICS):
                # synonyms ending with "@" or "_" expect the number k to follow, the others default to k=10
                query, k = (synonym + "5", 5) if synonym.endswith(("@", "_")) else (synonym, 10)
                yield metric_cls, query, metric_cls(k=k).key  # type: ignore[call-arg]
            else:
                yield metric_cls, synonym, metric_cls().key


@pytest.mark.parametrize(
    ("metric_cls", "query", "expected"),
    [
        pytest.param(
            *case,
            id=f"{case[0].__name__}-{case[1]}",
            marks=pytest.mark.xfail(strict=True, reason="synonym followed by an explicit k is not parsed")
            if (case[0].__name__, case[1]) in UNPARSED_SYNONYMS_WITH_K
            else (),
        )
        for case in _iter_rank_based_synonym_cases()
    ],
)
def test_rank_based_metric_synonyms_resolve(metric_cls: type[RankBasedMetric], query: str, expected: str):
    """Test that every synonym of a rank-based metric resolves to the key of the metric."""
    key = RankBasedMetricResults.key_from_string(query)
    assert isinstance(key, RankBasedMetricKey)
    assert key.metric == expected


# --- abstract classes ----------------------------------------------------------------------------------------------


def _import_all_modules() -> list[Any]:
    """Import all modules, ignoring those with missing optional dependencies."""
    modules = []
    for name in MODULE_NAMES:
        try:
            modules.append(importlib.import_module(name))
        except ModuleNotFoundError as error:
            if (error.name or "").startswith("pykeen"):
                raise
    return modules


def _is_abstract(value: Any) -> bool:
    """Return whether the class attribute is an abstract method."""
    if isinstance(value, classmethod | staticmethod):
        value = value.__func__
    if isinstance(value, property):
        value = value.fget
    return getattr(value, "__isabstractmethod__", False)


def _iter_classes_declaring_abstract_methods() -> Iterable[type]:
    """Iterate over all pykeen classes which declare an abstract method themselves."""
    seen = set()
    for module in _import_all_modules():
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls in seen or not cls.__module__.startswith("pykeen"):
                continue
            seen.add(cls)
            if any(_is_abstract(value) for value in vars(cls).values()):
                yield cls


# These classes declare abstract methods, but do not use `abc.ABCMeta` as metaclass; hence, instantiating them is
# possible although methods are missing.
NOT_ENFORCED_ABSTRACT_CLASSES = {
    "pykeen.evaluation.evaluation_loop.EvaluationLoop",
    "pykeen.evaluation.evaluator.MetricResults",
    "pykeen.losses.AdversarialLoss",
    "pykeen.losses.Loss",
    "pykeen.losses.PairwiseLoss",
    "pykeen.losses.PointwiseLoss",
    "pykeen.metrics.ranking.RankBasedMetric",
    "pykeen.nn.node_piece.tokenization.Tokenizer",
    "pykeen.nn.text.encoder.TextEncoder",
    "pykeen.nn.weighting.EdgeWeighting",
    "pykeen.predict.PredictionDataset",
    "pykeen.predict.ScoreConsumer",
    "pykeen.sampling.filtering.Filterer",
    "pykeen.sampling.negative_sampler.NegativeSampler",
    "pykeen.triples.splitting.Cleaner",
    "pykeen.triples.splitting.Splitter",
}


def _qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


@pytest.mark.parametrize(
    "cls",
    [
        pytest.param(
            cls,
            id=_qualified_name(cls),
            marks=pytest.mark.xfail(strict=True, reason="declares abstract methods, but is not an abstract base class")
            if _qualified_name(cls) in NOT_ENFORCED_ABSTRACT_CLASSES
            else (),
        )
        for cls in sorted(_iter_classes_declaring_abstract_methods(), key=_qualified_name)
    ],
)
def test_abstract_methods_are_enforced(cls: type):
    """Test that a class which declares abstract methods has a metaclass which enforces them."""
    assert isinstance(cls, abc.ABCMeta)


# --- repr ----------------------------------------------------------------------------------------------------------


def _iter_extra_repr_modules() -> Iterable[type[torch.nn.Module]]:
    """Iterate over all pykeen modules which use :class:`pykeen.utils.ExtraReprMixin`."""
    _import_all_modules()
    pending = list(ExtraReprMixin.__subclasses__())
    seen = set()
    while pending:
        cls = pending.pop()
        if cls in seen:
            continue
        seen.add(cls)
        pending.extend(cls.__subclasses__())
        if issubclass(cls, torch.nn.Module) and cls.__module__.startswith("pykeen"):
            yield cls


# The base classes which put the mixin behind `torch.nn.Module`, so that the latter's (empty) `extra_repr` is used.
EXTRA_REPR_BROKEN_BASES = (Representation, Combination, Decomposition)


@pytest.mark.parametrize(
    "cls",
    [
        pytest.param(
            cls,
            id=_qualified_name(cls),
            marks=pytest.mark.xfail(
                strict=True, reason="torch.nn.Module.extra_repr takes precedence over ExtraReprMixin.extra_repr"
            )
            if issubclass(cls, EXTRA_REPR_BROKEN_BASES)
            else (),
        )
        for cls in sorted(_iter_extra_repr_modules(), key=_qualified_name)
    ],
)
def test_extra_repr_mixin_is_used(cls: type[torch.nn.Module]):
    """Test that modules using the extra repr mixin do not get their extra repr overridden by torch's default."""
    assert cls.extra_repr is not torch.nn.Module.extra_repr


@pytest.mark.xfail(strict=True, reason="torch.nn.Module.extra_repr takes precedence over ExtraReprMixin.extra_repr")
def test_representation_repr_shows_parameters():
    """Test that the repr of a representation contains its parameters."""
    assert "max_id=3" in repr(Embedding(max_id=3, shape=4))


def test_extra_repr_mixin_provides_components_without_module():
    """Test that the mixin produces the components given by `iter_extra_repr`, independent of any module."""

    class _Example(ExtraReprMixin):
        def iter_extra_repr(self) -> Collection[str]:
            return ["a=1", "b=2"]

    assert repr(_Example()) == "_Example(a=1, b=2)"
