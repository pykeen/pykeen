# -*- coding: utf-8 -*-

"""Inductive models in PyKEEN."""

from .base import InductiveERModel
from .inductive_nodepiece import InductiveNodePiece
from .inductive_nodepiece_gnn import InductiveNodePieceGNN

__all__ = [
    "InductiveERModel",
    "InductiveNodePiece",
    "InductiveNodePieceGNN",
]
