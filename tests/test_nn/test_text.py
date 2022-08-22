"""Test textual encoders."""

import unittest_templates

import pykeen.nn.text
from tests import cases

from ..utils import needs_packages


class CharacterEmbeddingTextEncoderTestCase(cases.TextEncoderTestCase):
    """A test case for the character embedding based text encoder."""

    cls = pykeen.nn.text.CharacterEmbeddingTextEncoder


@needs_packages("transformers")
class TransformerTextEncoderTestCase(cases.TextEncoderTestCase):
    """A test case for the transformer encoder."""

    cls = pykeen.nn.text.TransformerTextEncoder


class TextEncoderMetaTestCase(unittest_templates.MetaTestCase):
    """Test for tests for text encoders."""

    base_cls = pykeen.nn.text.TextEncoder
    base_test = cases.TextEncoderTestCase
