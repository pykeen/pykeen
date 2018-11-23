PyKEEN |build| |docs| |zenodo|
==============================

PyKEEN (Python KnowlEdge EmbeddiNgs) is a package for training and evaluating knowledge graph embeddings.

The system has a modular architecture, and can be configured by the user through the command line interface.

Citation
--------
If you use BioKEEN in your work, please cite:

.. [1] Ali, M., *et al.* (2018). `BioKEEN: A library for learning and evaluating biological knowledge graph embeddings <https://doi.org/10.1101/475202>`_. bioRxiv 475202.

Installation |pypi_version| |python_versions| |pypi_license|
------------------------------------------------------------
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
To summarize the results of all experiments, please run following command:

.. code-block:: sh

    pykeen-summarize -d /path/to/experiments/directory -o /path/to/output/file.csv

or alternatively:

.. code-block:: python

    python3 -m pykeen-summarize -d /path/to/experiments/directory -o /path/to/output/file.csv

.. |build| image:: https://travis-ci.org/SmartDataAnalytics/PyKEEN.svg?branch=master
    :target: https://travis-ci.org/SmartDataAnalytics/PyKEEN

.. |zenodo| image:: https://zenodo.org/badge/136345023.svg
   :target: https://zenodo.org/badge/latestdoi/136345023

.. |docs| image:: http://readthedocs.org/projects/pykeen/badge/?version=latest
    :target: https://pykeen.readthedocs.io/en/latest/
    :alt: Documentation Status
.. |python_versions| image:: https://img.shields.io/pypi/pyversions/pykeen.svg
    :alt: Stable Supported Python Versions
.. |pypi_version| image:: https://img.shields.io/pypi/v/pykeen.svg
    :alt: Current version on PyPI
.. |pypi_license| image:: https://img.shields.io/pypi/l/pykeen.svg
    :alt: MIT License
