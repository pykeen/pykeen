# -*- coding: utf-8 -*-

"""POEM."""

import logging

from .version import get_version  # noqa: F401

# This will set the global logging level to info to ensure that info messages are shown in all parts of the software.
logging.getLogger(__name__).setLevel(logging.INFO)
