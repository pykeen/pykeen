Train and Evaluate
==================
Configure your experiment
~~~~~~~~~~~~~~~~~~~~~~~~~
To programmatically train (and evaluate) a KGE model, a python dictionary must be created specifying the experiment:

.. code-block:: python

    config = dict(
        training_set_path           = 'data/corpora/fb15k/fb_15k_train.tsv',
        execution_mode              = 'Training_mode',
        kg_embedding_model_name     = 'TransE',
        embedding_dim               = 50,
        normalization_of_entities   = 2,  # corresponds to L2
        scoring_function            = 1,  # corresponds to L1
        margin_loss                 = 1,
        learning_rate               = 0.01,
        batch_size                  = 32,
        num_epochs                  = 1000,
        test_set_ratio              = 0.1,
        filter_negative_triples     = True,
        random_seed                 = 2,
        preferred_device            = 'gpu',
    )

The ``training_set_path`` can also be set to a list of strings. The corresponding knowledge graphs will all be combine
before training.

.. code-block:: python

    config = dict(
        training_set_path = [
            'data/corpora/fb15k/fb_15k_train.tsv',
            'data/corpora/fb16k/fb_16k_train.tsv',
            ...
        ],
        ...
    )

Run your experiment
~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    results = pykeen.run(
        config=config,
        output_directory=output_directory,
    )

Access your results
~~~~~~~~~~~~~~~~~~~
Show all keys contained in ``results``:

.. code-block:: python

    print('Keys:', *sorted(results.results.keys()), sep='\n  ')

Access trained KGE model
~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    results.trained_model

Access the losses
~~~~~~~~~~~~~~~~~~
.. code-block:: python

    results.losses

Access evaluation results
~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    results.evaluation_summary

