"""Unimodal KGE Models."""

from .auto_sf import AutoSF
from .boxe import BoxE
from .compgcn import CompGCN
from .complex import ComplEx
from .conv_e import ConvE
from .conv_kb import ConvKB
from .cp import CP
from .crosse import CrossE
from .distma import DistMA
from .distmult import DistMult
from .ermlp import ERMLP
from .ermlpe import ERMLPE
from .hole import HolE
from .kg2e import KG2E
from .mure import MuRE
from .node_piece import NodePiece
from .ntn import NTN
from .pair_re import PairRE
from .proj_e import ProjE
from .quate import QuatE
from .rescal import RESCAL
from .rgcn import RGCN
from .rotate import RotatE
from .simple import SimplE
from .structured_embedding import SE
from .toruse import TorusE
from .trans_d import TransD
from .trans_e import TransE
from .trans_f import TransF
from .trans_h import TransH
from .trans_r import TransR
from .tucker import TuckER
from .unstructured_model import UM

__all__ = [
    "AutoSF",
    "BoxE",
    "CompGCN",
    "ComplEx",
    "ConvE",
    "ConvKB",
    "CP",
    "CrossE",
    "DistMA",
    "DistMult",
    "ERMLP",
    "ERMLPE",
    "HolE",
    "KG2E",
    "MuRE",
    "NTN",
    "NodePiece",
    "PairRE",
    "ProjE",
    "QuatE",
    "RESCAL",
    "RGCN",
    "RotatE",
    "SimplE",
    "SE",
    "TorusE",
    "TransD",
    "TransE",
    "TransF",
    "TransH",
    "TransR",
    "TuckER",
    "UM",
]
