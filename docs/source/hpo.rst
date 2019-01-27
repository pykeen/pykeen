Apply a Hyper-Parameter Optimization
====================================

Here, we describe how to define an experiment that should perform a hyper-parameter optimization mode.


Configure your experiment
~~~~~~~~~~~~~~~~~~~~~~~~~
To run PyKEEN in hyper-parameter optimization (HPO) mode, please set **execution_mode**  to **HPO_mode**.
In HPO mode several values can be provided for the hyper-parameters from which different settings will be tested based
on the hyper-parameter optimization algorithm. The possible values for a single hyper-parameter need to be provided as
a list. The **maximum_number_of_hpo_iters** defines how many HPO iterations should be performed.

.. code-block:: python

    config = dict(
        training_set_path           = 'data/corpora/fb15k/fb_15k_train.tsv',
        test_set_path               = 'data/corpora/fb15k/fb_15k_test.tsv',
        execution_mode              = 'HPO_mode',
        kg_embedding_model_name     = 'TransE',
        embedding_dim               = [50,100,150]
        normalization_of_entities   =  2,  # corresponds to L2
        scoring_function            = [1,2],  # corresponds to L1
        margin_loss                 = [1,1.5,2],
        learning_rate               = [0.1,0.01],
        batch_size                  = [32,128],
        num_epochs                  = 1000,
        maximum_number_of_hpo_iters = 3,
        filter_negative_triples     = True,
        random_seed                 = 2,
        preferred_device            = 'gpu',
    )


Run your experiment
~~~~~~~~~~~~~~~~~~~
The experiment will be started with the *run* function, and in the output directory the exported results will be saved.

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

    results.results['trained_model']

Access the losses
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    results.results['losses']

Access evaluation results
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    results.results['eval_summary']

