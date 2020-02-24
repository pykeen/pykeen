Perform Inference
=================
PyKEEN can be used to perform inference based on a trained KGE model. While several approaches to realise the
inference workflow are conceivable, in our approach, users provide a set of candidate entities and relations that are
used to create all triple-permutations for which predictions are computed. Furthermore, users can provide a set of
triples that will be automatically removed from the set of candidate triples. This might be relevant in a setting,
in which predictions for all possible triples expect those contained in the training set should be computed.
The output of the inference workflow is a ranked list of triples where the most plausible ones are located
at the beginning of the list.

.. image:: ../images/inference.png

Starting the Prediction Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: sh

   pykeen-predict -m /path/to/model/directory -d /path/to/data/directory

where the value for the argument **-m** is the directory containing the model, in more detail following files must be
contained in the directory:

* configuration.json
* entities_to_embeddings.json
* relations_to_embeddings.json
* trained_model.pkl

These files are created automatically created after a model is trained (and evaluated) and exported in your
specified output directory.

The value for the argument **-d** is the directory containing the data for which inference should be applied, and it
needs to contain following files:

* entities.tsv
* relations.tsv

where *entities.tsv* contains all entities of interest, and relations.tsv all relations. Both files should contain
should contain a single column containing all the entities/relations. Based on these files, PyKEEN will create all
possible combinations of triples, and computes the predictions for them, and saves them in data directory
in *predictions.tsv*.
Note: the model- and the data-directory can be the same directory as long as all required files are provided.

Optionally, a set of triples can be provided that should be exluded from the prediction, e.g. all the triples
contained in the training set:

.. code-block:: sh

   pykeen-predict -m /path/to/model/directory -d /path/to/data/directory -t /path/to/triples.tsv

Hence, it is easily possible to compute plausibility scores for all triples that are not contained in the training set.

CLI Manual
~~~~~~~~~~
.. click:: pykeen.cli.cli:predict
   :prog: pykeen-predict
   :show-nested:
