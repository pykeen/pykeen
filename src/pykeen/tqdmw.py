# -*- coding: utf-8 -*-

"""A wrapper around tqdm to make it automatically play nice with Jupyter Notebook."""

__all__ = [
    'tqdm',
    'trange',
]


def is_notebook() -> bool:
    """Check if we're running in a Jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if is_notebook():
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange
