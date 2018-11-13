Knowledge Graph Embedding Models
================================

This page contains all the Knowledge Graph Embedding Models included in PyKEEN.

+------------------------+---------------------+
| Model Name             | Reference           |
|                        |                     |
+========================+=====================+
| TransE                 | Bordes_ *et al.*    |
+------------------------+---------------------+
| TransH                 | Wang_ *et al.*    |
+------------------------+---------------------+
| TransR                 | Lin_ *et al.*    |
+------------------------+---------------------+
| TransD                 | Ji_ *et al.*    |
+------------------------+---------------------+
| ConvE                  | Dettermers_ *et al.*    |
+------------------------+---------------------+
| SE                     | Bordes2_ *et al.*    |
+------------------------+---------------------+
| UM                     | Bordes3_ *et al.*    |
+------------------------+---------------------+
| RESCAL                 | Nickel_ *et al.*    |
+------------------------+---------------------+
| ERMLP                  | Dong_ *et al.*    |
+------------------------+---------------------+
| DistMult               | Yang_ *et al.*    |
+------------------------+---------------------+

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
.. _Jin: http://www.aclweb.org/anthology/P15-1067 
.. _Dettmers: https://arxiv.org/pdf/1707.01476.pdf
.. _Bordes2: http://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/download/3659/3898 
.. _Bordes3: https://link.springer.com/content/pdf/10.1007%2Fs10994-013-5363-6.pdf 
.. _Nickel: http://www.cip.ifi.lmu.de/~nickel/data/slides-icml2011.pdf 
.. _Dong: https://dl.acm.org/citation.cfm?id=2623623
.. _Yang: https://arxiv.org/pdf/1412.6575.pdf
