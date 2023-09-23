# -*- coding: utf-8 -*-

"""Version information for PyKEEN."""

import os
import sys
from functools import lru_cache
from subprocess import CalledProcessError, check_output  # noqa: S404
from typing import Optional, Tuple

__all__ = [
    "VERSION",
    "get_version",
    "get_git_hash",
    "get_git_branch",
    "env",
]

VERSION = "1.10.2-dev"


@lru_cache(maxsize=2)
def get_git_hash(terse: bool = True) -> str:
    """Get the PyKEEN git hash.

    :param terse: Should the hash be clipped to 8 characters?
    :return:
        The git hash, equals 'UNHASHED' if encountered CalledProcessError, signifying that the
        code is not installed in development mode.
    """
    rv = _run("git", "rev-parse", "HEAD")
    if rv is None:
        return "UNHASHED"
    if terse:
        return rv[:8]
    return rv


@lru_cache(maxsize=1)
def get_git_branch() -> Optional[str]:
    """Get the PyKEEN branch, if installed from git in editable mode.

    :return:
        Returns the name of the current branch, or None if not installed in development mode.
    """
    return _run("git", "branch", "--show-current")


def _run(*args: str) -> Optional[str]:
    with open(os.devnull, "w") as devnull:
        try:
            ret = check_output(  # noqa: S603,S607
                args,
                cwd=os.path.dirname(__file__),
                stderr=devnull,
            )
        except (CalledProcessError, FileNotFoundError):
            return None
        else:
            return ret.strip().decode("utf-8")


def get_version(with_git_hash: bool = False) -> str:
    """Get the PyKEEN version string, including a git hash.

    :param with_git_hash:
        If set to True, the git hash will be appended to the version.
    :return: The PyKEEN version as well as the git hash, if the parameter with_git_hash was set to true.
    """
    return f"{VERSION}-{get_git_hash(terse=True)}" if with_git_hash else VERSION


def env_table(tablefmt: str = "github", headers: Tuple[str, str] = ("Key", "Value")) -> str:
    """Generate a table describing the environment in which PyKEEN is being run."""
    import platform
    import time

    import torch
    from tabulate import tabulate

    rows = [
        ("OS", os.name),
        ("Platform", platform.system()),
        ("Release", platform.release()),
        ("Time", str(time.asctime())),
        ("Python", f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"),
        ("PyKEEN", get_version()),
        ("PyKEEN Hash", get_git_hash()),
        ("PyKEEN Branch", get_git_branch()),
        ("PyTorch", torch.__version__),
        ("CUDA Available?", str(torch.cuda.is_available()).lower()),
        ("CUDA Version", torch.version.cuda or "N/A"),
        ("cuDNN Version", torch.backends.cudnn.version() or "N/A"),
    ]
    return tabulate(rows, tablefmt=tablefmt, headers=headers)


def env_html():
    """Output the environment table as HTML for usage in Jupyter."""
    from IPython.display import HTML

    return HTML(env_table(tablefmt="html"))


def env(file=None):
    """Print the env or output as HTML if in Jupyter.

    :param file: The file to print to if not in a Jupyter setting. Defaults to sys.stdout
    :returns: A :class:`IPython.display.HTML` if in a Jupyter notebook setting, otherwise none.
    """
    if _in_jupyter():
        return env_html()
    else:
        print(env_table(), file=file)  # noqa:T201


def _in_jupyter() -> bool:
    try:
        get_ipython = sys.modules["IPython"].get_ipython  # type: ignore
        if "IPKernelApp" not in get_ipython().config:
            raise ImportError("console")
        if "VSCODE_PID" in os.environ:
            raise ImportError("vscode")
    except Exception:
        return False
    else:
        return True


if __name__ == "__main__":
    print(get_version(with_git_hash=True))  # noqa:T201
