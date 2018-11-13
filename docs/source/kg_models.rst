Knowledge Graph Embedding Models
================================

This page contains all the Knowledge Graph Embedding Models included in PyKEEN.

+------------------------+---------------------+
| Model Name             | Reference           |
|                        |                     |
+========================+=====================+
| TransE                 | |transE|            |
+------------------------+---------------------+
| TransH                 | |transH|            |
+------------------------+---------------------+
| TransR                 | |transR|            |
+------------------------+---------------------+
| TransD                 | |transD|            |
+------------------------+---------------------+
| ConvE                  | |convE|             |
+------------------------+---------------------+
| SE                     | |SE|                |
+------------------------+---------------------+
| UM                     | |UM|                |
+------------------------+---------------------+
| RESCAL                 | |RESCAL|            |
+------------------------+---------------------+
| ERMLP                  | |ERMLP|             |
+------------------------+---------------------+
| DistMult               | |DistMult|          |
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
