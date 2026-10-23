"""Generic tests for the default HPO search spaces (``hpo_default``) of all components that define one.

The tests are parametrized over every model, loss, regularizer, optimizer, learning rate scheduler, negative sampler and
training loop, and check that

* the search space only refers to parameters that the component accepts,
* every required parameter has a search space,
* every entry of the search space is understood by :func:`pykeen.hpo.hpo.suggest_kwargs`, i.e., is sampled at all,
* a log-scale entry is actually sampled on a log scale, and
* components can be constructed from configurations at the bounds of their search space.

Known violations are marked as strict expected failures, so that fixing one of them requires removing its mark.
"""

import inspect
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

import numpy as np
import optuna
import pytest
import torch
from optuna.distributions import BaseDistribution, CategoricalDistribution
from optuna.trial import FixedTrial, Trial

from pykeen.hpo.hpo import suggest_kwargs
from pykeen.losses import loss_resolver
from pykeen.lr_schedulers import lr_scheduler_resolver, lr_schedulers_hpo_defaults
from pykeen.models import model_resolver
from pykeen.optimizers import optimizer_resolver, optimizers_hpo_defaults
from pykeen.regularizers import regularizer_resolver
from pykeen.sampling import negative_sampler_resolver
from pykeen.training import TrainingLoop, training_loop_resolver
from pykeen.triples import TriplesNumericLiteralsFactory
from pykeen.triples.generation import generate_triples_factory

optuna.logging.set_verbosity(optuna.logging.WARNING)

#: the keys of a search space entry which are understood by `suggest_kwargs`
KNOWN_SPEC_KEYS = {"type", "low", "high", "log", "scale", "q", "step", "base", "choices"}

#: the scales which are understood by `suggest_kwargs`, for integer parameters
KNOWN_INT_SCALES = {None, "linear", "power", "power_two"}

#: the number of entities and relations of the (tiny) triples factory models are constructed with
NUM_ENTITIES = 13
NUM_RELATIONS = 5

# ------------------------------------------------------------------------------------------------------------------
# Collect the components
# ------------------------------------------------------------------------------------------------------------------


class Component:
    """A component with a search space."""

    def __init__(
        self,
        kind: str,
        name: str,
        cls: type,
        hpo_default: Mapping[str, Mapping[str, Any]],
        signature_target: Callable[..., Any],
        provided: Collection[str],
        make: Callable[[Mapping[str, Any]], object] | None,
    ) -> None:
        """
        Initialize the component.

        :param kind: the kind of component, e.g., "model"
        :param name: the (normalized) name of the component
        :param cls: the component class
        :param hpo_default: the default search space
        :param signature_target: the callable whose parameters the search space is expected to refer to
        :param provided: the names of required parameters which are not part of the search space, since they are
            provided by the surrounding code, e.g., the triples factory for models
        :param make: a callable to create an instance from a configuration, or None, if the configuration is not
            used for construction
        """
        self.kind = kind
        self.name = name
        self.cls = cls
        self.hpo_default = hpo_default
        self.signature_target = signature_target
        self.provided = set(provided)
        self.make = make

    @property
    def id(self) -> str:
        """Return an ID for the test case."""
        return f"{self.kind}-{self.name}"

    @property
    def prefix(self) -> str:
        """Return the prefix used for parameter names."""
        return self.kind.replace("-", "_")


def _triples_factory() -> TriplesNumericLiteralsFactory:
    """Create a tiny triples factory with numeric literals, which is suitable for all (multi-modal) models."""
    core = generate_triples_factory(
        num_entities=NUM_ENTITIES, num_relations=NUM_RELATIONS, num_triples=50, random_state=0
    )
    return TriplesNumericLiteralsFactory(
        mapped_triples=core.mapped_triples,
        entity_to_id={f"e{i}": i for i in range(NUM_ENTITIES)},
        relation_to_id={f"r{i}": i for i in range(NUM_RELATIONS)},
        numeric_literals=np.random.default_rng(0).uniform(size=(NUM_ENTITIES, 3)),
        literals_to_id={f"l{i}": i for i in range(3)},
    )


def _collect_components() -> list[Component]:
    components: list[Component] = []

    def _make_model(cls: type) -> Callable[[Mapping[str, Any]], object]:
        def _make(config: Mapping[str, Any]) -> object:
            extra = {}
            if cls.__module__.startswith("pykeen.models.inductive"):
                extra["inference_factory"] = _triples_factory()
            return model_resolver.make(cls, config, triples_factory=_triples_factory(), **extra)

        return _make

    for name, cls in sorted(model_resolver.lookup_dict.items()):
        components.append(
            Component(
                "model",
                name,
                cls,
                cls.hpo_default,
                cls.__init__,
                {"triples_factory", "inference_factory"},
                _make_model(cls),
            )
        )

    for name, cls in sorted(loss_resolver.lookup_dict.items()):
        components.append(Component("loss", name, cls, cls.hpo_default, cls.__init__, (), lambda c, cls=cls: cls(**c)))

    for name, cls in sorted(regularizer_resolver.lookup_dict.items()):
        components.append(
            Component("regularizer", name, cls, cls.hpo_default, cls.__init__, (), lambda c, cls=cls: cls(**c))
        )

    def _make_optimizer(cls: type) -> Callable[[Mapping[str, Any]], object]:
        return lambda config: optimizer_resolver.make(cls, config, params=[torch.nn.Parameter(torch.zeros(1))])

    for optimizer_cls, hpo_default in optimizers_hpo_defaults.items():
        components.append(
            Component(
                "optimizer",
                optimizer_cls.__name__.lower(),
                optimizer_cls,
                hpo_default,
                optimizer_cls.__init__,
                {"params"},
                _make_optimizer(optimizer_cls),
            )
        )

    def _make_lr_scheduler(cls: type) -> Callable[[Mapping[str, Any]], object]:
        def _make(config: Mapping[str, Any]) -> object:
            optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
            return lr_scheduler_resolver.make(cls, config, optimizer=optimizer)

        return _make

    for lr_scheduler_cls, hpo_default in lr_schedulers_hpo_defaults.items():
        components.append(
            Component(
                "lr_scheduler",
                lr_scheduler_cls.__name__.lower(),
                lr_scheduler_cls,
                hpo_default,
                lr_scheduler_cls.__init__,
                {"optimizer"},
                _make_lr_scheduler(lr_scheduler_cls),
            )
        )

    def _make_sampler(cls: type) -> Callable[[Mapping[str, Any]], object]:
        return lambda config: negative_sampler_resolver.make(
            cls, config, mapped_triples=_triples_factory().mapped_triples
        )

    for name, cls in sorted(negative_sampler_resolver.lookup_dict.items()):
        components.append(
            Component(
                "negative_sampler", name, cls, cls.hpo_default, cls.__init__, {"mapped_triples"}, _make_sampler(cls)
            )
        )

    # note: the training parameters are not passed to the constructor, but to `train`
    for name, cls in sorted(training_loop_resolver.lookup_dict.items()):
        components.append(
            Component("training", name, cls, cls.hpo_default, TrainingLoop.train, {"triples_factory"}, None)
        )
    return components


COMPONENTS = _collect_components()

# ------------------------------------------------------------------------------------------------------------------
# Known violations
# ------------------------------------------------------------------------------------------------------------------

_LOG_SCALE_IGNORED = (
    "`suggest_kwargs` only reads `log` for float parameters, not `scale`, so `scale: log` is ignored and the "
    "parameter is sampled on a linear scale"
)

#: known violations by test name and test ID
XFAIL: dict[str, dict[str, str]] = {
    "test_required_parameters_covered": {
        "regularizer-combined": (
            "`CombinedRegularizer` requires `regularizers`, which has no entry in its search space, "
            "so HPO cannot construct it"
        ),
    },
    "test_log_scale_is_honored": dict.fromkeys(
        (
            "model-kg2e-c_min",
            "regularizer-combined-total_weight",
            "regularizer-lp-weight",
            "regularizer-normlimit-weight",
            "regularizer-orthogonality-weight",
            "regularizer-powersum-weight",
            "optimizer-adagrad-lr",
            "optimizer-adam-lr",
            "optimizer-adamax-lr",
            "optimizer-adamw-lr",
            "optimizer-sgd-lr",
            "lr_scheduler-cycliclr-base_lr",
            "lr_scheduler-cycliclr-max_lr",
            "lr_scheduler-onecyclelr-max_lr",
        ),
        _LOG_SCALE_IGNORED,
    ),
    "test_construct_from_bounds": {
        "model-distmultliteralgated": (
            "the model passes `entity_embedding_dim` to `GatedCombination`, which does not accept it, so it cannot be "
            "constructed with any configuration"
        ),
        "regularizer-combined": "`CombinedRegularizer` requires `regularizers`, which is not part of the search space",
        "lr_scheduler-multisteplr": (
            "`milestones` is sampled as a single integer, but the scheduler expects a sequence of integers"
        ),
        "lr_scheduler-onecyclelr": (
            "`OneCycleLR` requires `total_steps` or `epochs` and `steps_per_epoch`, none of which is in the search "
            "space"
        ),
    },
}


def _params(test_name: str, components: Sequence[Component] = COMPONENTS) -> list:
    """Create the test parameters, with expected failures marked."""
    xfail = XFAIL.get(test_name, {})
    result = []
    for component in components:
        id_ = component.id
        marks: tuple[Any, ...] = ()
        if id_ in xfail:
            marks = (pytest.mark.xfail(strict=True, reason=xfail[id_]),)
        result.append(pytest.param(component, id=id_, marks=marks))
    return result


# ------------------------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------------------------


def _sample(component: Component) -> tuple[Mapping[str, Any], Mapping[str, BaseDistribution]]:
    """Sample a configuration from the search space with a random trial."""
    trial = optuna.create_study().ask()
    config = suggest_kwargs(trial=trial, prefix=component.prefix, kwargs_ranges=component.hpo_default)
    return config, trial.distributions


def _bound_configs(component: Component) -> list[Mapping[str, Any]]:
    """Create configurations in which all parameters are at the lower / upper end of their ranges."""
    _, distributions = _sample(component)
    configs = []
    for pick in (0, -1):
        params: dict[str, Any] = {}
        for name, distribution in distributions.items():
            if isinstance(distribution, CategoricalDistribution):
                params[name] = distribution.choices[pick]
            else:
                params[name] = distribution.low if pick == 0 else distribution.high  # type: ignore[attr-defined]
        trial: Trial = FixedTrial(params)  # type: ignore[assignment]
        configs.append(suggest_kwargs(trial=trial, prefix=component.prefix, kwargs_ranges=component.hpo_default))
    return configs


# ------------------------------------------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("component", _params("test_keys_are_parameters"))
def test_keys_are_parameters(component: Component):
    """Test that the search space only refers to parameters the component accepts."""
    parameters = inspect.signature(component.signature_target).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        pytest.skip("the signature accepts arbitrary keyword arguments")
    assert not set(component.hpo_default).difference(parameters)


@pytest.mark.parametrize("component", _params("test_required_parameters_covered"))
def test_required_parameters_covered(component: Component):
    """Test that each required parameter has a search space."""
    parameters = inspect.signature(component.signature_target).parameters
    required = {
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}
    }
    required -= {"self"}
    required -= component.provided
    assert not required.difference(component.hpo_default)


@pytest.mark.parametrize("component", _params("test_specs_are_understood"))
def test_specs_are_understood(component: Component):
    """Test that every entry of the search space is sampled."""
    for name, spec in component.hpo_default.items():
        assert not set(spec).difference(KNOWN_SPEC_KEYS), (
            f"{name}: unknown keys {set(spec).difference(KNOWN_SPEC_KEYS)}"
        )
        if spec["type"] in {int, "int"}:
            assert spec.get("scale") in KNOWN_INT_SCALES, f"{name}: unknown scale {spec.get('scale')}"
    config, _ = _sample(component)
    assert set(config) == set(component.hpo_default)


def _log_scale_entries() -> list:
    """Create test parameters for all entries which request a log scale, one per entry."""
    result = []
    xfail = XFAIL.get("test_log_scale_is_honored", {})
    for component in COMPONENTS:
        for name, spec in component.hpo_default.items():
            if spec.get("scale") != "log" and spec.get("log") not in {True, "true"}:
                continue
            id_ = f"{component.id}-{name}"
            marks: tuple[Any, ...] = ()
            if id_ in xfail:
                marks = (pytest.mark.xfail(strict=True, reason=xfail[id_]),)
            result.append(pytest.param(component, name, id=id_, marks=marks))
    return result


@pytest.mark.parametrize(("component", "name"), _log_scale_entries())
def test_log_scale_is_honored(component: Component, name: str):
    """Test that a parameter which is requested to be sampled on a log scale is sampled on a log scale."""
    _, distributions = _sample(component)
    distribution = distributions[f"{component.prefix}.{name}"]
    assert getattr(distribution, "log", False)


@pytest.mark.parametrize("component", [p for p in _params("test_construct_from_bounds") if p.values[0].make])
def test_construct_from_bounds(component: Component):
    """Test that the component can be constructed from configurations at the bounds of the search space."""
    assert component.make is not None
    for config in _bound_configs(component):
        assert component.make(config) is not None
