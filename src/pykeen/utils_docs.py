# -*- coding: utf-8 -*-

"""Utilities for documentation."""

import textwrap
from typing import TypeVar

import yaml

__all__ = [
    'with_structured_docstr',
]

X = TypeVar('X')


def with_structured_docstr(obj: X, delimiter: str = '---') -> X:
    """Parse the structured data from the end of the docstr and store it in ``__docdata__``.

    The data after the delimiter should be in the YAML form.
    It is parsed with :func:`yaml.safe_load` then stored in the ``__docdata__`` field of the
    object.

    :param obj: Any object that can has a ``__doc__`` field.
    :param delimiter: The delimiter between the actual docstring and structured YAML.
    :return: The same object with a modified docstr.

    :raises AttributeError: if the object has no ``__doc__`` field.
    """
    try:
        docstr = obj.__doc__
    except AttributeError:
        raise AttributeError(f'no __doc__ available in {obj}')
    if docstr is None:  # no docstr to modify
        return obj

    lines = docstr.splitlines()
    try:
        index = min(
            i
            for i, line in enumerate(lines)
            if line.strip() == delimiter
        )
    except ValueError:
        return obj

    # The docstr is all of the lines before the line with the delimiter. No
    # modification to the text wrapping is necessary.
    obj.__doc__ = '\n'.join(lines[:index])

    # The YAML structured data is on all lines following the line with the delimiter.
    # The text must be dedented before YAML parsing.
    yaml_str = textwrap.dedent('\n'.join(lines[index + 1:]))
    obj.__docdata__ = yaml.safe_load(yaml_str)  # type: ignore
    return obj
