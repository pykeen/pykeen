Using Tensorboard
=========================

`Tensorboard <https://www.tensorflow.org/tensorboard/>`_ (TB) is a service for tracking experimental results during training.
It is part of the Tensorflow project.

Minimal Pipeline Example
---------------------------------
A CSV log file can be generated with the following:

.. code-block:: python

    from pykeen.pipeline import pipeline

    pipeline_result = pipeline(
        model='RotatE',
        dataset='Kinships',
        result_tracker='tensorboard',
    )

It is placed in a subdirectory of :mod:`pystow` default data directory with PyKEEN called ``tensorboard``,
which will likely be at ``~/.data/pykeen/logs/tensorboard`` on your system. The file is named based on the
current time if no alternative is provided.

Specifying a Name
-----------------
If you want to specify the name of the log file in the default directory, use the ``experiment_name`` keyword
argument like:

.. code-block:: python

    from pykeen.pipeline import pipeline

    pipeline_result = pipeline(
        model='RotatE',
        dataset='Kinships',
        result_tracker='tensorboard',
        result_tracker_kwargs=dict(
        experiment_name='rotate-kinships',
        ),
    )

Specifying a Custom Log Directory
-----------------
If you want to specify a custom directory to store the tensorboard logs, use the ``experiment_path`` keyword
argument like:

.. code-block:: python

    from pykeen.pipeline import pipeline

    pipeline_result = pipeline(
        model='RotatE',
        dataset='Kinships',
        result_tracker='tensorboard',
        result_tracker_kwargs=dict(
        experiment_path='tb-logs/rotate-kinships',
        ),
    )

Please be aware that if you re-run an experiment using the same directory, then the logs will be combined.
It is advisable to use a unique sub-directory for each experiment to allow for easy comparison.
