# -*- coding: utf-8 -*-

"""Exceptions for the pipeline."""

from textwrap import dedent

from class_resolver import KeywordArgumentError

__all__ = [
    'ModelArgumentError',
]


class ModelArgumentError(TypeError):
    """Thrown when the pipeline is unable to build a model because of a missing kwarg."""

    def __init__(self, e: KeywordArgumentError):
        self.e = e

    def __str__(self) -> str:
        return dedent(f"""\
        {self.e}

        Please specify the '{self.e.name}' as a key in the 'model_kwargs' dictionary passed to pipeline().
        """)
