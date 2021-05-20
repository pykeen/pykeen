# -*- coding: utf-8 -*-

"""Test for sLCWA and LCWA."""

from pykeen.sampling.filtering import BloomFilterer, PythonSetFilterer
from pykeen.training import LCWATrainingLoop, SLCWATrainingLoop
from tests.test_training import cases


class UnfilteredSLCWATrainingLoopTestCase(cases.SLCWATrainingLoopTestCase):
    """Test sLCWA with unfiltered negative sampling."""

    cls = SLCWATrainingLoop
    filterer = None


class SetFilteredSLCWATrainingLoopTestCase(cases.SLCWATrainingLoopTestCase):
    """Test sLCWA with set filtered negative sampling."""

    cls = SLCWATrainingLoop
    filterer = PythonSetFilterer


class BloomFilteredSLCWATrainingLoopTestCase(cases.SLCWATrainingLoopTestCase):
    """Test sLCWA with bloom filtered negative sampling."""

    cls = SLCWATrainingLoop
    filterer = BloomFilterer


class LCWATrainingLoopTestCase(cases.TrainingLoopTestCase):
    """Test LCWA."""

    cls = LCWATrainingLoop
