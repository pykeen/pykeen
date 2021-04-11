# -*- coding: utf-8 -*-

"""Version information for PyKEEN."""

import os
import sys
from subprocess import CalledProcessError, check_output  # noqa: S404

__all__ = [
    'VERSION',
    'get_version',
    'get_git_hash',
]

VERSION = '1.4.1-dev'


def get_git_hash() -> str:
    """Get the PyKEEN git hash.

    :return:
        The git hash, equals 'UNHASHED' if encountered CalledProcessError, signifying that the
        code is not installed in development mode.
    """
    with open(os.devnull, 'w') as devnull:
        try:
            ret = check_output(  # noqa: S603,S607
                ['git', 'rev-parse', 'HEAD'],
                cwd=os.path.dirname(__file__),
                stderr=devnull,
            )
        except CalledProcessError:
            return 'UNHASHED'
        else:
            return ret.strip().decode('utf-8')[:8]


def get_version(with_git_hash: bool = False) -> str:
    """Get the PyKEEN version string, including a git hash.

    :param with_git_hash:
        If set to True, the git hash will be appended to the version.
    :return: The PyKEEN version as well as the git hash, if the parameter with_git_hash was set to true.
    """
    return f'{VERSION}-{get_git_hash()}' if with_git_hash else VERSION


def env_table(tablefmt='github', headers=('Key', 'Value')) -> str:
    """Generate a table describing the environment in which PyKEEN is being run."""
    import torch
    import platform
    from tabulate import tabulate
    t1 = [
        ('`os.name`', os.name),
        ('`platform.system()`', platform.system()),
        ('`platform.release()`', platform.release()),
        ('python', f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}'),
        ('pykeen', get_version(with_git_hash=True)),
        ('torch', torch.__version__),
        ('cuda available', str(torch.cuda.is_available()).lower()),
        ('cuda', torch.version.cuda),
        ('cudnn', torch.backends.cudnn.version()),
    ]
    return tabulate(t1, tablefmt=tablefmt, headers=headers)


if __name__ == '__main__':
    print(get_version(with_git_hash=True))
