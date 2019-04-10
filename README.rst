PyKEEN |build| |coverage| |docs| |zenodo|
=========================================

PyKEEN (Python KnowlEdge EmbeddiNgs) is a package for training and evaluating knowledge graph embeddings. Currently,
it provides implementations of 10 knowledge graph emebddings models, and can be run in *training mode* in which users
provide their own set of hyper-parameter values, or in *hyper-parameter optimization mode* to find suitable
hyper-parameter values from set of user defined values. PyKEEN can also be run without having experience in programing
by using its interactive command line interface that can be started with the command *pykeen* from a terminal.

Share Your Experimental Artifacts
---------------------------------
You can share you trained KGE models along the other experimental artifacts through the `KEEN-Model-Zoo <https://github.com/SmartDataAnalytics/KEEN-Model-Zoo>`_.


Installation |pypi_version| |python_versions| |pypi_license|
------------------------------------------------------------
``pykeen`` can be installed on any system running Python 3.6+ with the following command:

.. code-block:: sh

    pip install pykeen

Alternatively, it can be installed from the source for development with:

.. code-block:: sh

    $ git clone https://github.com/SmartDataAnalytics/PyKEEN.git pykeen
    $ cd pykeen
    $ pip install -e .

However, GPU acceleration is limited to Linux systems with the appropriate graphics cards
as described in the PyTorch documentation.

Installation Extras
~~~~~~~~~~~~~~~~~~~
PyKEEN has several extras that can be installed with PyPI.

1. ``pip install pykeen[rdf]`` enables support for loading files using ``rdflib``. They can be added to the training
   file paths by prefixing files with ``rdf:``
2. ``pip install pykeen[ndex]`` enables support for loading networks from NDEx. They can be added to the training file
   paths by prefixing files with ``ndex:``

Any combinations can be added with commas, such as ``pip install pykeen[rdf,ndex]``.

Tutorials
---------
Code examples can be found in the `notebooks directory
<https://github.com/SmartDataAnalytics/PyKEEN/tree/master/notebooks>`_.

Further tutorials are available in our `documentation <https://pykeen.readthedocs.io/en/latest/>`_.

CLI Usage - Set Up Your Experiment within 60 seconds
----------------------------------------------------
To start the PyKEEN CLI, run the following command:

.. code-block:: sh

    pykeen

then the command line interface will assist you to configure your experiments.

To start PyKEEN with an existing configuration file, run:

.. code-block:: sh

    pykeen -c /path/to/config.json

then the command line interface won't be called, instead the pipeline will be started immediately.

Starting the Prediction Pipeline
********************************
To make prediction based on a trained model, run:

.. code-block:: sh

    pykeen-predict -m /path/to/model/directory -d /path/to/data/directory

where the value for the argument **-m** is the directory containing the model, in more detail following files must be
contained in the directory:

* configuration.json
* entities_to_embeddings.json
* relations_to_embeddings.json
* trained_model.pkl

These files are created automatically created after model is trained (and evaluated) and exported in your
specified output directory.

The value for the argument **-d** is the directory containing the data for which inference should be applied, and it
needs to contain following files:

* entities.tsv
* relations.tsv

where *entities.tsv* contains all entities of interest, and relations.tsv all relations. Both files should contain
should contain a single column containing all the entities/relations. Based on these files, PyKEEN will create all
triple permutations, and computes the predictions for them, and saves them in data directory
in *predictions.tsv*.
Note: the model- and the data-directory can be the same directory as long as all required files are provided.

Optionally, a set of triples can be provided that should be exluded from the prediction, e.g. all the triples
contained in the training set:

.. code-block:: sh

   pykeen-predict -m /path/to/model/directory -d /path/to/data/directory -t /path/to/triples.tsv

Hence, it is easily possible to compute plausibility scores for all triples that are not contained in the training set.

Summarize the Results of All Experiments
****************************************
To summarize the results of all experiments, please provide the path to parent directory containing all the experiments
as sub-directories, and the path to the output file:

.. code-block:: sh

    pykeen-summarize -d /path/to/experiments/directory -o /path/to/output/file.csv

.. |build| image:: https://travis-ci.org/SmartDataAnalytics/PyKEEN.svg?branch=master
    :target: https://travis-ci.org/SmartDataAnalytics/PyKEEN
    :alt: Build Status

.. |zenodo| image:: https://zenodo.org/badge/136345023.svg
    :target: https://zenodo.org/badge/latestdoi/136345023
    :alt: Zenodo DOI

.. |docs| image:: http://readthedocs.org/projects/pykeen/badge/?version=latest
    :target: https://pykeen.readthedocs.io/en/latest/
    :alt: Documentation Status

.. |python_versions| image:: https://img.shields.io/pypi/pyversions/pykeen.svg
    :alt: Supported Python Versions: 3.6 and 3.7

.. |pypi_version| image:: https://img.shields.io/pypi/v/pykeen.svg
    :alt: Current version on PyPI

.. |pypi_license| image:: https://img.shields.io/pypi/l/pykeen.svg
    :alt: MIT License

.. |coverage| image:: https://codecov.io/gh/SmartDataAnalytics/PyKEEN/branch/master/graphs/badge.svg
    :target: https://codecov.io/gh/SmartDataAnalytics/PyKEEN
    :alt: Coverage Status on CodeCov
