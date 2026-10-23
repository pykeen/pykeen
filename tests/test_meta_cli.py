"""Meta-tests for the command line interfaces.

They discover all click commands defined in PyKEEN modules, and check that the help of each command and subcommand
can be rendered, and that groups with a default subcommand refer to a command that exists.
"""

import importlib
import inspect
import pkgutil
from collections.abc import Iterable

import click
import click_default_group
import pytest
from _pytest.mark.structures import ParameterSet
from click.testing import CliRunner

import pykeen
from pykeen.losses import loss_resolver


def _iter_modules() -> Iterable[tuple[str, object]]:
    """Import all PyKEEN modules which can be imported, skipping those missing optional dependencies."""
    for info in sorted(pkgutil.walk_packages(pykeen.__path__, prefix="pykeen."), key=lambda info: info.name):
        try:
            yield info.name, importlib.import_module(info.name)
        except ModuleNotFoundError as error:
            if error.name is None or error.name.split(".")[0] == "pykeen":
                raise


def _iter_subcommands(command: click.Command) -> Iterable[click.Command]:
    """Iterate recursively over all subcommands."""
    if isinstance(command, click.Group):
        for subcommand in command.commands.values():
            yield subcommand
            yield from _iter_subcommands(subcommand)


def _collect_root_commands() -> dict[str, click.Command]:
    """Find all click commands which are defined at the top level of a PyKEEN module, and are not a subcommand."""
    found: dict[int, tuple[str, click.Command]] = {}
    for module_name, module in _iter_modules():
        for attribute, value in vars(module).items():
            if not isinstance(value, click.Command):
                continue
            name = f"{module_name}:{attribute}"
            # the same command can be imported in several modules (e.g., into ``__main__``)
            old = found.get(id(value))
            if old is None or ("__main__" in old[0], old[0]) > ("__main__" in name, name):
                found[id(value)] = (name, value)
    subcommand_ids = {id(sub) for _, command in found.values() for sub in _iter_subcommands(command)}
    return dict(sorted(name_and_command for key, name_and_command in found.items() if key not in subcommand_ids))


ROOT_COMMANDS = _collect_root_commands()


def _iter_command_paths(
    command: click.Command, path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], click.Command]]:
    """Iterate over the paths (names to invoke) of a command and all its subcommands, together with the commands."""
    yield path, command
    if isinstance(command, click.Group):
        for name, subcommand in sorted(command.commands.items()):
            yield from _iter_command_paths(subcommand, (*path, name))


def _click_choice_accepts_context() -> bool:
    """Check whether the choice type created by class-resolver can render its metavar with click's arguments.

    Since click 8.2, ``get_metavar`` is called with the context. If class-resolver's choice type does not accept it,
    rendering the help of every command with a resolver-based choice option raises a :class:`TypeError`.
    """
    choice = loss_resolver._get_click_choice()
    return "ctx" in inspect.signature(choice.get_metavar).parameters


_CHOICE_IS_BROKEN = not _click_choice_accepts_context()
_CHOICE_BROKEN_REASON = (
    "the choice type of class-resolver does not accept the context argument which click passes to get_metavar, "
    "so rendering the help raises a TypeError"
)


def _uses_resolver_choice(command: click.Command) -> bool:
    """Check whether a command has an option or argument with a choice type created by class-resolver."""
    return any(type(param.type).__qualname__.startswith("BaseResolver._get_click_choice") for param in command.params)


def _iter_help_cases() -> Iterable[ParameterSet]:
    for root, root_command in ROOT_COMMANDS.items():
        for path, command in _iter_command_paths(root_command):
            marks = []
            if _CHOICE_IS_BROKEN and _uses_resolver_choice(command):
                marks.append(pytest.mark.xfail(strict=True, raises=TypeError, reason=_CHOICE_BROKEN_REASON))
            yield pytest.param(root, path, marks=marks, id=" ".join((root, *path)))


@pytest.mark.parametrize(("root", "path"), _iter_help_cases())
def test_help(root: str, path: tuple[str, ...]) -> None:
    """Test that the help of each command and subcommand can be rendered."""
    result = CliRunner().invoke(ROOT_COMMANDS[root], [*path, "--help"])
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def _iter_default_group_cases() -> Iterable[ParameterSet]:
    for root, root_command in ROOT_COMMANDS.items():
        for path, group in _iter_command_paths(root_command):
            if not isinstance(group, click_default_group.DefaultGroup):
                continue
            marks = []
            # `pykeen ls` is a default group with the (non-existing) default command `github-readme`
            if (root, path) == ("pykeen.cli:main", ("ls",)):
                marks.append(
                    pytest.mark.xfail(
                        strict=True, reason="the default subcommand `github-readme` of `pykeen ls` does not exist"
                    )
                )
            yield pytest.param(group, marks=marks, id=" ".join((root, *path)))


@pytest.mark.parametrize("group", _iter_default_group_cases())
def test_default_subcommand_exists(group: click_default_group.DefaultGroup) -> None:
    """Test that the default subcommand of a group exists, so invoking the group without arguments works."""
    assert group.default_cmd_name in group.commands
