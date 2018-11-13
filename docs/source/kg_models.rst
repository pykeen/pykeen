Knowledge Graph Embedding Models
================================

This page contains all the Knowledge Graph Embedding Models included in PyKEEN.

+------------------------+---------------------+
| Model Name             | Reference           |
|                        |                     |
+========================+=====================+
| TransE                 | Bordes_ *et al.*    |
+------------------------+---------------------+
| TransH                 | Wang_ *et al.*      |
+------------------------+---------------------+
| TransR                 | Lin_ *et al.*       |
+------------------------+---------------------+
| TransD                 | Ji_ *et al.*        |
+------------------------+---------------------+
| ConvE                  | Dettermers_ *et al.*|
+------------------------+---------------------+
| SE                     | Bordes2_ *et al.*   |
+------------------------+---------------------+
| UM                     | Bordes3_ *et al.*   |
+------------------------+---------------------+
| RESCAL                 | Nickel_ *et al.*    |
+------------------------+---------------------+
| ERMLP                  | Dong_ *et al.*      |
+------------------------+---------------------+
| DistMult               | Yang_ *et al.*      |
+------------------------+---------------------+

References
----------

- Bordes, A., *et al.* (2013). Translating embeddings for modeling multi-relational data. NIPS..
- Wang, Z., *et al.* (2014). Knowledge Graph Embedding by Translating on Hyperplanes. AAAI. Vol. 14.
- Lin, Y., *et al.* (2015). Learning entity and relation embeddings for knowledge graph completion. AAAI. Vol. 15.
- Ji, G., *et al.* (2015). Knowledge graph embedding via dynamic mapping matrix. ACL.
- Dettmers, T., *et al.* (2017) Convolutional 2d knowledge graph embeddings. arXiv preprint arXiv:1707.01476.
- Bordes, A., *et al.* (2011). Learning Structured Embeddings of Knowledge Bases. AAAI. Vol. 6. No. 1.
- Bordes, A., *et al.* (2014). A semantic matching energy function for learning with multi-relational data.
 Machine Learning 94.2 : 233-259.
- Nickel, M., *et al.* (2011) A Three-Way Model for Collective Learning on Multi-Relational Data. ICML. Vol. 11.
- Dong, X., *et al.* (2014) Knowledge vault: A web-scale approach to probabilistic knowledge fusion. ACM.
- Yang, B. *et al.* Embedding entities and relations for learning and inference in knowledge bases. arXiv preprint
 arXiv:1412.6575

TransE
------
Considers a relation as a translation from the head to the tail entity.

.. automodule:: pykeen.kg_embeddings_model.trans_e
    :members:

TransH
------
Extends TransE by applying the translation from head to tail entity in a relational-specific hyperplane.

.. automodule:: pykeen.kg_embeddings_model.trans_h
    :members:

TransR
------
Extends TransE and TransH by considering different vector spaces for entities and relations.

.. automodule:: pykeen.kg_embeddings_model.trans_r
    :members:

TransD
------
Extends TransR to use fewer parameters.

.. automodule:: pykeen.kg_embeddings_model.trans_d
    :members:

DistMul
-------
Simplifies RESCAL by restricting matrices representing relations as diagonal matrices.

.. automodule:: pykeen.kg_embeddings_model.distmult
    :members:

ConvE
-----
Uses a convolutional neural network (CNN).

.. automodule:: pykeen.kg_embeddings_model.conv_e
    :members:

ERMPL
-----
Neural network based approach.

.. automodule:: pykeen.kg_embeddings_model.ermlp
    :members:

RESCAL
------
Represents relations as matrices and models interactions between latent features.

.. automodule:: pykeen.kg_embeddings_model.rescal
    :members:

Structured Embedding (SE)
-------------------------
For each relation head and tail entity are projected by different matrices.

.. automodule:: pykeen.kg_embeddings_model.structured_embedding
    :members:

Unstructured Model (UM)
-------------------------
.. automodule:: pykeen.kg_embeddings_model.unstructured_model
    :members:

.. _Bordes: http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf
.. _Wang: https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546
.. _Lin: http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523/
.. _Ji: http://www.aclweb.org/anthology/P15-1067
.. _Dettermers: https://arxiv.org/pdf/1707.01476.pdf
.. _Bordes2: http://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/download/3659/3898
.. _Bordes3: https://link.springer.com/content/pdf/10.1007%2Fs10994-013-5363-6.pdf
.. _Nickel: http://www.cip.ifi.lmu.de/~nickel/data/slides-icml2011.pdf
.. _Dong: https://dl.acm.org/citation.cfm?id=2623623
.. _Yang: https://arxiv.org/pdf/1412.6575.pdf
