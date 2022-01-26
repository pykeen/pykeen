"""Tests for node piece."""
import unittest_templates

import pykeen.nn.node_piece
from tests import cases


class DegreeAnchorSelectionTestCase(cases.AnchorSelectionTestCase):
    """Tests for degree anchor selection."""

    cls = pykeen.nn.node_piece.DegreeAnchorSelection


class AnchorSelectionMetaTestCase(unittest_templates.MetaTestCase[pykeen.nn.node_piece.AnchorSelection]):
    """Test for tests for anchor selection strategies."""

    base_cls = pykeen.nn.node_piece.AnchorSelection
    base_test = cases.AnchorSelectionTestCase
