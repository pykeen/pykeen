KEEN
====

KEEN (KnowlEdge EmbeddiNgs) is a package for training and evaluating knowledge graph embeddings.
The system has a modular architecture, and can be configured by the user through the command line interface.
 
**Currently, the framework is under heavy development.**

Installation
------------
1. ``keen`` can be installed with the following commmands:

.. code-block:: sh

    python3 -m pip install git+https://github.com/SmartDataAnalytics/KEEN.git@master

2. or in editable mode with:

.. code-block:: sh

    git clone https://github.com/SmartDataAnalytics/KEEN.git

.. code-block:: sh

    cd keen

.. code-block:: sh

    python3 -m pip install -e .
    
## Project Structure

* In 'src/deployer' all deployer scripts are contained.
* In 'src/kg_embeddings_model' all knowledge graph embedding models are defined.
* In 'src/utilities' all utilities such as building the pipelines.
