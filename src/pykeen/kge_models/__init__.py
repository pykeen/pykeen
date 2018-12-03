# -*- coding: utf-8 -*-

"""Implementations of various knowledge graph embedding models."""

from pykeen.kge_models.conv_e import ConvE  # noqa: F401
from pykeen.kge_models.distmult import DistMult  # noqa: F401
from pykeen.kge_models.ermlp import ERMLP  # noqa: F401
from pykeen.kge_models.rescal import RESCAL  # noqa: F401
from pykeen.kge_models.structured_embedding import StructuredEmbedding  # noqa: F401
from pykeen.kge_models.trans_d import TransD  # noqa: F401
from pykeen.kge_models.trans_e import TransE  # noqa: F401
from pykeen.kge_models.trans_h import TransH  # noqa: F401
from pykeen.kge_models.trans_r import TransR  # noqa: F401
from pykeen.kge_models.unstructured_model import UnstructuredModel  # noqa: F401
from pykeen.kge_models.utils import get_kge_model  # noqa: F401
