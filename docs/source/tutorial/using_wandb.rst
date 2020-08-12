Using Weights and Biases
========================
`Weights and Biases <http://wandb.ai/>`_ (WANDB) is a service for tracking experimental results and various artifacts appearing
whn training ML models.

Below are the necessary steps to start using WANDB in PyKEEN:

* To use WANDB, you first need to register on the website
* Obtain your ``api_key`` which can be found in profile Settings.
* Install WANDB ``pip install wandb``
* Perform login with your unique api key ``wandb login <api_key>``
* Create a project in WANDB, for example, with a name ``pykeen_experiments``

Now you can simply specify this project name when initializing a pipeline, and everything else will work automatically!

Pipeline Example
----------------
This example shows using WANDB with the :func:`pykeen.pipeline.pipeline` function.

.. code-block:: python

    from pykeen.pipeline import pipeline

    results = pipeline(
        model='RotatE',
        dataset='Kinships',
        result_tracker='wandb'
        result_tracker_kwargs=dict(
            project_name='pykeen_experiments',
        ),
    )


You can navigate to the created project in WANDB and observe a running experiment.
Further tweaking of appearance, charts, and other settings is described in the official `documentation <https://docs.wandb.com/>`_

HPO Example
-----------
This example shows using WANDB with the :func:`pykeen.hpo.hpo_pipeline` function.

.. code-block:: python

    from pykeen.hpo import hpo_pipeline

    results = hpo_pipeline(
        model='RotatE',
        dataset='Kinships',
        result_tracker='mlflow'
        result_tracker_kwargs=dict(
            project_name='pykeen_experiments',
            experiment_name='new run'
        ),
    )


You can also specify an optional ``experiment_name`` which will appear on the website instead of randomly generated labels.


Additional documentation of the valid keyword arguments can be found
under :class:`pykeen.trackers.WANDBResultTracker`.