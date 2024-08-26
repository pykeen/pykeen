"""Tests for checkpointing."""

from typing import ClassVar

from unittest_templates import MetaTestCase

from pykeen.checkpoints.keeper import CheckpointKeeper
from tests.cases import CheckpointKeeperBase


class CheckpointKeeperMetaTestCase(MetaTestCase[CheckpointKeeper]):
    """Meta test case for checkpoint keepers."""

    base_cls: ClassVar = CheckpointKeeper
    base_test: ClassVar = CheckpointKeeperBase
