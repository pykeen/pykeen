PyKEEN |build|
============
PyKEEN (Python KnowlEdge EmbeddiNgs) is a package for training and evaluating knowledge graph embeddings.

The system has a modular architecture, and can be configured by the user through the command line interface.

**Currently, the framework is under heavy development.**

Installation
------------
1. ``PyKEEN`` can be installed with the following commands:

.. code-block:: sh

    python3 -m pip install git+https://github.com/SmartDataAnalytics/PyKEEN.git@master

2. or in editable mode with:

.. code-block:: sh

    $ git clone https://github.com/SmartDataAnalytics/PyKEEN.git keen
    $ cd pykeen
    $ python3 -m pip install -e .

How to Use
----------
To start PyKEEN, please run the following command:

.. code-block:: sh

    pykeen

or alternatively:

.. code-block:: python

    python3 -m pykeen

then the command line interface will assist you to configure your experiments.

.. |build| image:: https://travis-ci.org/SmartDataAnalytics/PyKEEN.svg?branch=master
    :target: https://travis-ci.org/SmartDataAnalytics/PyKEEN
