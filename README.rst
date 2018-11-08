PyKEEN |build||zenodo|  
======================

PyKEEN (Python KnowlEdge EmbeddiNgs) is a package for training and evaluating knowledge graph embeddings.

The system has a modular architecture, and can be configured by the user through the command line interface.

Installation
------------
1. ``PyKEEN`` can be installed with the following commands:

.. code-block:: sh

    python3 -m pip install git+https://github.com/SmartDataAnalytics/PyKEEN.git@master

2. or in editable mode with:

.. code-block:: sh

    $ git clone https://github.com/SmartDataAnalytics/PyKEEN.git pykeen
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

To start PyKEEN with an existing configuration file, please run the following command:

.. code-block:: sh

    pykeen -c /path/to/config.json

or alternatively:

.. code-block:: python

    python3 -m pykeen -c /path/to/config.json

then the command line interface won't be called, instead the pipeline will be started immediately.

Starting PyKEEN's prediction pipeline
**************************************
To make prediction based on a trained model, please run following command:

.. code-block:: sh

    pykeen-predict -m /path/to/model/directory -d /path/to/data/directory

or alternatively:

.. code-block:: python

    python3 -m pykeen-predict -m /path/to/model/directory -d /path/to/data/directory

Summarize the results of all experiments
****************************************
To summarize the results of all experiments, please switch to root directory containing the directories for each
experiment, and run following command:

.. code-block:: sh

    pykeen-summarize

or alternatively:

.. code-block:: python

    python3 -m pykeen-summarize

.. |build| image:: https://travis-ci.org/SmartDataAnalytics/PyKEEN.svg?branch=master
    :target: https://travis-ci.org/SmartDataAnalytics/PyKEEN
