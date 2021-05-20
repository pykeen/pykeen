# -*- coding: utf-8 -*-

"""Test for sLCWA and LCWA."""

from pykeen.training import LCWATrainingLoop, SLCWATrainingLoop
from tests.test_trainer import cases


class FilteredSLCWATrainingLoopTestCase(cases.SLCWATrainingLoopTestCase):
    """Test sLCWA with filtered negative sampling."""

    cls = SLCWATrainingLoop
    filtered = True


class UnfilteredSLCWATrainingLoopTestCase(cases.SLCWATrainingLoopTestCase):
    """Test sLCWA with unfiltered negative sampling."""

    cls = SLCWATrainingLoop
    filtered = False


class LCWATrainingLoopTestCase(cases.TrainingLoopTestCase):
    """Test LCWA."""

    cls = LCWATrainingLoop
