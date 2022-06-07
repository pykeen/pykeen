"""Test textual encoders."""
import unittest

import unittest_templates

import pykeen.nn.text
from tests import cases

try:
    import transformers
except ImportError:
    transformers = None


class CharacterEmbeddingTextEncoderTestCase(cases.TextEncoderTestCase):
    """A test case for the character embedding based text encoder."""

    cls = pykeen.nn.text.CharacterEmbeddingTextEncoder


@unittest.skipIf(transformers is None, "Need to install `transformers`")
class TransformerTextEncoderTestCase(cases.TextEncoderTestCase):
    """A test case for the transformer encoder."""

    cls = pykeen.nn.text.TransformerTextEncoder


class TextEncoderMetaTestCase(unittest_templates.MetaTestCase):
    """Test for tests for text encoders."""

    base_cls = pykeen.nn.text.TextEncoder
    base_test = cases.TextEncoderTestCase
