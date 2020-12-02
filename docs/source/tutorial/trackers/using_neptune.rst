Using Neptune.ai
================
`Neptune <https://neptune.ai>`_ is a graphical tool for tracking the results of machine learning. PyKEEN integrates
Neptune into the pipeline and HPO pipeline.

Preparation
-----------
1. To use it, you'll first have to install Neptune's client with ``pip install neptune-client`` or
   install PyKEEN with the ``neptune`` extra with ``pip install pykeen[neptune]``.
2. Create an account at `Neptune <https://neptune.ai>`_.

   - Get an API token following `this tutorial <https://docs.neptune.ai/security-and-privacy/api-tokens/how-to-find-and-set-neptune-api-token.html>`_.
   - [Optional] Set the ``NEPTUNE_API_TOKEN`` environment variable to your API token.
3. [Optional] Create a new project by following `this tutorial for project and user
   management <https://docs.neptune.ai/workspace-project-and-user-management/projects/create-project.html>`_.
   Neptune automatically creates a project for all new users called ``sandbox`` which you
   can directly use.

Pipeline Example
----------------
This example shows using Neptune with the :func:`pykeen.pipeline.pipeline` function.
Minimally, the ``project_qualified_name`` and ``experiment_name`` must be set.

.. code-block:: python

    from pykeen.pipeline import pipeline

    pipeline_result = pipeline(
        model='RotatE',
        dataset='Kinships',
        result_tracker='neptune',
        result_tracker_kwargs=dict(
            project_qualified_name='cthoyt/sandbox',
            experiment_name='Tutorial Training of RotatE on Kinships',
        ),
    )

.. warning::

    If you haven't set the ``NEPTUNE_API_TOKEN`` environment variable, the ``api_token`` becomes
    a mandatory key.

Reusing Experiments
-------------------
In the Neptune web application, you'll see that experiments are assigned an ID. This means you can re-use the same
ID to group different sub-experiments together using the ``experiment_id`` keyword argument instead of
``experiment_name``.

.. code-block:: python

    from pykeen.pipeline import pipeline

    experiment_id = 4  # if doesn't already exist, will throw an error!
    pipeline_result = pipeline(
        model='RotatE',
        dataset='Kinships',
        result_tracker='neptune'
        result_tracker_kwargs=dict(
            project_qualified_name='cthoyt/sandbox',
            experiment_id=4,
        ),
    )

Don't worry - you can keep using the ``experiment_name`` argument and the experiment's identifier will
be automatically looked up eah time.

Adding Tags
-----------
Tags are additional information that you might want to add to the experiment
and store in Neptune. Note this is different from MLflow, which considers tags
as key/value pairs.

For example, if you're using custom input, you might want to add some labels
about if the experiment is cool or not.

.. code-block:: python

    from pykeen.pipeline import pipeline

    data_version = ...

    pipeline_result = pipeline(
        model='RotatE',
        training=...,
        testing=...,
        validation=...,
        result_tracker='mlflow',
        result_tracker_kwargs=dict(
            project_qualified_name='cthoyt/sandbox',
            experiment_name='Tutorial Training of RotatE on Kinships',
            tags={'cool', 'doggo'},
        ),
    )

Additional documentation of the valid keyword arguments can be found
under :class:`pykeen.trackers.NeptuneResultTracker`.
