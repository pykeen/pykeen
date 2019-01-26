Use The CLI To Perform Inference
================================

Starting the Prediction Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: sh

   pykeen-predict -m /path/to/model/directory -d /path/to/data/directory

where the value for the argument **-m** is the directory containing the model, in more detail following files must be contained in
the directory:

* configuration.json
* entities_to_embeddings.json
* relations_to_embeddings.json
* trained_model.pkl

These files are created automatically created when an experiment is configured through the CLI.

The vlue for the argument  **-d** is the directory containing the data for which inference should be applied, and it needs
to contain following files:

* entities.tsv
* relations.tsv

where *entities.tsv* contains all entities of interest, and relations.tsv all relations. PyKEEN will create all possible
combinations of triples, and computes the predictions for them, and saves them in data directory in *predictions.tsv*.

Optionally, a set of triples can be provided that should be exluded from the prediction, e.g. all the triples
contained in the training set:

.. code-block:: sh

   pykeen-predict -m /path/to/model/directory -d /path/to/data/directory -t /path/to/triples.tsv

Hence, it is easily possible to compute plausibility scores forr all triples that are not contained in the training set.