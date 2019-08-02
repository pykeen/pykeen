# -*- coding: utf-8 -*-

"""Unimodal KGE Models.

.. [bordes2011] Bordes, A., *et al.* (2011). `Learning Structured Embeddings of Knowledge Bases
                <http://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/download/3659/3898>`_. AAAI. Vol. 6. No. 1.
.. [bordes2013] Bordes, A., *et al.* (2013). `Translating embeddings for modeling multi-relational data
                <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_.
                NIPS.
.. [bordes2014] Bordes, A., *et al.* (2014). `A semantic matching energy function for learning with
                multi-relational data <https://link.springer.com/content/pdf/10.1007%2Fs10994-013-5363-6.pdf>`_.
                Machine
.. [dong2014] Dong, X., *et al.* (2014) `Knowledge vault: A web-scale approach to probabilistic knowledge fusion
              <https://dl.acm.org/citation.cfm?id=2623623>`_. ACM.
.. [ji2015] Ji, G., *et al.* (2015). `Knowledge graph embedding via dynamic mapping matrix
            <http://www.aclweb.org/anthology/P15-1067>`_. ACL.
.. [kazemi2018] S. M. Kazemi, D. Poole (2018). `SimplE Embedding for Link Prediction in Knowledge Graphs`
            <https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs>_. NIPS'18
.. [lin2015] Lin, Y., *et al.* (2015). `Learning entity and relation embeddings for knowledge graph completion
             <http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523/>`_. AAAI. Vol. 15.
.. [nguyen2018] Nguyen, D. Q., *et al* (2018) `A Novel Embedding Model for Knowledge Base CompletionBased on
                Convolutional Neural Network <https://www.aclweb.org/anthology/N18-2053>`_.
                *NAACL-HLT 2018*
.. [nickel2011] Nickel, M., *et al.* (2011) `A Three-Way Model for Collective Learning on Multi-Relational Data
                <http://www.cip.ifi.lmu.de/~nickel/data/paper-icml2011.pdf>`_. ICML. Vol. 11.

.. [nickel2016] Nickel, M. *et al.* (2016) `Holographic Embeddings of Knowledge Graphs
                <https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828>`_. AAAI 2016.
.. [sun2019] Sun, Z., *et al.* (2019) `RotatE: Knowledge Graph Embeddings by relational rotation in complex space
             <https://arxiv.org/pdf/1902.10197v1.pdf>`_. ICLR 2019.
.. [trouillon2016] Trouillon, T., *et al.* (2016) Complex embeddings for simple link prediction.
                   International Conference on Machine Learning. 2016.
.. [wang2014] Wang, Z., *et al.* (2014). `Knowledge Graph Embedding by Translating on Hyperplanes
                  <https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546>`_. AAAI. Vol. 14.
.. [yang2014] Yang, B., *et al.* (2014). `Embedding Entities and Relations for Learning
                  and Inference in Knowledge Bases <https://arxiv.org/pdf/1412.6575.pdf>`_. CoRR, abs/1412.6575.
.. [socher2013] Socher, R., *et al.* (2013) `Reasoning with neural tensor networks for knowledge base completion. <https://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-networks-for-knowledge-base-completion>`_ NIPS. 2013.
"""

from .complex import ComplEx
from .conv_kb import ConvKB
from .distmult import DistMult
from .ermlp import ERMLP
from .hole import HolE
from .ntn import NTN
from .rescal import RESCAL
from .rotate import RotatE
from .simple import SimplE
from .structured_embedding import StructuredEmbedding
from .trans_d import TransD
from .trans_e import TransE
from .trans_h import TransH
from .trans_r import TransR
from .unstructured_model import UnstructuredModel

__all__ = [
    'ConvKB',
    'ComplEx',
    'DistMult',
    'ERMLP',
    'HolE',
    'NTN',
    'RESCAL',
    'SimplE',
    'StructuredEmbedding',
    'TransD',
    'TransE',
    'TransH',
    'TransR',
    'RotatE',
    'UnstructuredModel',
]
