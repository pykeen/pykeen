Using MLflow
============
`MLflow <https://mlflow.org>`_ is a graphical tool for tracking the results of machine learning. PyKEEN integrates
MLflow into the pipeline and HPO pipeline.

To use it, you'll first have to install MLflow with ``pip install mlflow`` and run it in the background
with ``mlflow ui``. More information can be found on the
`MLflow Quickstart <https://mlflow.org/docs/latest/quickstart.html>`_. It'll be running at http://localhost:5000
by default.

Pipeline Example
----------------
This example shows using MLflow with the :func:`pykeen.pipeline.pipeline` function.

.. code-block:: python

    from pykeen.pipeline import pipeline

    results = pipeline(
        model='RotatE',
        dataset='Kinships',
        mlflow_tracking_uri='http://localhost:5000',
        mlflow_experiment_name='Tutorial Training of RotatE on Kinships',
    )

If you navigate to the MLflow UI at http://localhost:5000, you'll see the experiment appeared
in the left column.

.. image:: ../img/mlflow_tutorial_1.png
  :alt: MLflow home

If you click on the experiment, you'll see this:

.. image:: ../img/mlflow_tutorial_2.png
  :alt: MLflow experiment view

HPO Example
-----------
This example shows using MLflow with the :func:`pykeen.hpo.hpo_pipeline` function.

.. code-block:: python

    from pykeen.hpo import hpo_pipeline

    results = hpo_pipeline(
        model='RotatE',
        dataset='Kinships',
        mlflow_tracking_uri='http://localhost:5000',
        mlflow_experiment_name='Tutorial HPO Training of RotatE on Kinships',
    )

The same navigation through MLflow can be done for this example.

Reusing Experiments
-------------------
In the MLflow UI, you'll see that experiments are assigned an ID. This means you can re-use the same ID to group
different sub-experiments together using the ``mlflow_experiment_id`` keyword argument instead of
``mlflow_experiment_name``.

.. code-block:: python

    from pykeen.pipeline import pipeline

    experiment_id = 4  # if doesn't already exist, will throw an error!
    results = pipeline(
        model='RotatE',
        dataset='Kinships',
        mlflow_tracking_uri='http://localhost:5000',
        mlflow_experiment_id=4,
    )
