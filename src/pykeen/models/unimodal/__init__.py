# -*- coding: utf-8 -*-

"""Unimodal KGE Models.

.. [balazevic2019] Balažević, *et al.* (2019) `TuckER: Tensor Factorization for Knowledge Graph Completion
                   <https://arxiv.org/abs/1901.09590>`_. EMNLP'19
.. [bordes2011] Bordes, A., *et al.* (2011). `Learning Structured Embeddings of Knowledge Bases
                <http://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/download/3659/3898>`_. AAAI. Vol. 6. No. 1.
.. [bordes2013] Bordes, A., *et al.* (2013). `Translating embeddings for modeling multi-relational data
                <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_.
                NIPS.
.. [bordes2014] Bordes, A., *et al.* (2014). `A semantic matching energy function for learning with
                multi-relational data <https://link.springer.com/content/pdf/10.1007%2Fs10994-013-5363-6.pdf>`_.
                Machine
.. [dettmers2018] Dettmers, T., *et al.* (2018) `Convolutional 2d knowledge graph embeddings
                <https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17366>`_.
                Thirty-Second AAAI Conference on Artificial Intelligence.
.. [dong2014] Dong, X., *et al.* (2014) `Knowledge vault: A web-scale approach to probabilistic knowledge fusion
              <https://dl.acm.org/citation.cfm?id=2623623>`_. ACM.
.. [ebisu2018] Ebisu, T., *et al.* (2018) `https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16227
               <https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16227>`_. AAAI'18.
.. [feng2016] Feng, J. *et al.* (2016) `Knowledge Graph Embedding by Flexible Translation
              <https://www.aaai.org/ocs/index.php/KR/KR16/paper/view/12887>`_. KR'16.
.. [ji2015] Ji, G., *et al.* (2015). `Knowledge graph embedding via dynamic mapping matrix
            <http://www.aclweb.org/anthology/P15-1067>`_. ACL.
.. [kazemi2018] Kazemi, S.M. and Poole, D. (2018). `SimplE Embedding for Link Prediction in Knowledge Graphs
                <https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs>`_. NIPS'18
.. [he2015] Shizhu, H., *et al.* (2017). `Learning to Represent Knowledge Graphs with Gaussian Embedding
            <http://ir.ia.ac.cn/bitstream/173211/20634/1/Learning%20to%20Represent%20Knowledge%20Graphs%20with%20Gaussian%20Embedding.pdf>`_.
            CIKM'17.
.. [lin2015] Lin, Y., *et al.* (2015). `Learning entity and relation embeddings for knowledge graph completion
             <http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523/>`_. AAAI. Vol. 15.
.. [nguyen2018] Nguyen, D. Q., *et al* (2018) `A Novel Embedding Model for Knowledge Base CompletionBased on
                Convolutional Neural Network <https://www.aclweb.org/anthology/N18-2053>`_.
                *NAACL-HLT 2018*
.. [nickel2011] Nickel, M., *et al.* (2011) `A Three-Way Model for Collective Learning on Multi-Relational Data
                <http://www.cip.ifi.lmu.de/~nickel/data/paper-icml2011.pdf>`_. ICML. Vol. 11.
.. [nickel2016] Nickel, M. *et al.* (2016) `Holographic Embeddings of Knowledge Graphs
                <https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828>`_. AAAI 2016.
.. [schlichtkrull2018] Schlichtkrull, M., *et al.* (2018) `Modeling relational data with graph convolutional networks
                       <https://arxiv.org/pdf/1703.06103>`_. ESWC'18.
.. [sharifzadeh2019] Sharifzadeh *et al.* (2019) Extension of ERMLP in PyKEEN.
.. [shi2017] Shi, B., and Weninger, T. `ProjE: Embedding Projection for Knowledge Graph Completion
             <https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14279>`_, AAAI 2017
.. [sun2019] Sun, Z., *et al.* (2019) `RotatE: Knowledge Graph Embeddings by relational rotation in complex space
             <https://arxiv.org/abs/1902.10197v1>`_. ICLR 2019.
.. [trouillon2016] Trouillon, T., *et al.* (2016) Complex embeddings for simple link prediction.
                   International Conference on Machine Learning. 2016.
.. [wang2014] Wang, Z., *et al.* (2014). `Knowledge Graph Embedding by Translating on Hyperplanes
              <https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546>`_. AAAI. Vol. 14.
.. [yang2014] Yang, B., *et al.* (2014). `Embedding Entities and Relations for Learning
              and Inference in Knowledge Bases <https://arxiv.org/abs/1412.6575>`_. CoRR, abs/1412.6575.
.. [socher2013] Socher, R., *et al.* (2013) `Reasoning with neural tensor networks for knowledge base completion.
                <https://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-networks-for-knowledge-base-completion>`_.
                NIPS. 2013.
.. [shi2019] Shi, X. *et al.* (2019). `Modeling Multi-mapping Relations for Precise Cross-lingual Entity Alignment
             <https://www.aclweb.org/anthology/D19-1075>`_. EMNLP-IJCNLP 2019.
.. [vashishth2020] Vashishth, S., *et al.* (2020). `Composition-based multi-relational graph convolutional
   networks <https://arxiv.org/abs/1911.03082>`_. *arXiv*, 1–15.
.. [zhang2019] Zhang, Shuai, *et al.* (2019). `Quaternion knowledge graph embeddings
                <https://openreview.net/forum?id=cZbk98eY_WwC>`_ NeurIPS'19.
.. [zhang2019b] Zhang, W., *et al.* (2019). `Interaction Embeddings for Prediction and Explanation in Knowledge
   Graphs <https://doi.org/10.1145/3289600.3291014>`. WSDM '19: Proceedings of the Twelfth ACM International
   Conference on Web Search and Data Mining.
.. [abboud2020] Abboud, R., *et al.* (2020). `BoxE: A box embedding model for knowledge base completion
   <https://proceedings.neurips.cc/paper/2020/file/6dbbe6abe5f14af882ff977fc3f35501-Paper.pdf>`_.
   *Advances in Neural Information Processing Systems*, 2020-December(NeurIPS), 1–13.
.. [galkin2021] Galkin, M., *et al.* (2021) `NodePiece: Compositional and Parameter-Efficient Representations
   of Large Knowledge Graphs <https://arxiv.org/abs/2106.12144>`_. *arXiv*, 2106.12144.
.. [zaheer2017] Zaheer, M., *et al.* (2017). `Deep sets
   <https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html>`_.
   *Advances in Neural Information Processing Systems*, 2017-December(ii), 3392–3402.
.. [lacroix2018] Lacroix, T., Usunier, N., & Obozinski, G. (2018). `Canonical Tensor Decomposition for Knowledge Base
   Completion <http://arxiv.org/abs/1806.07297>`_. *arXiv*, 1806.07297.
.. [hitchcock1927] Hitchcock, F. L. `The expression of a tensor or a polyadic as a sum of
   products <https://doi.org/10.1002/sapm192761164>`_. Studies in Applied Mathematics, 6 (1-4):164–189, 1927.
"""

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
