Using File-Based Tracking
=========================
Rather than logging to an external backend like MLflow or W&B, file based trackers
write to local files.

Minimal Pipeline Example with CSV
---------------------------------
A CSV log file can be generated with the following:

.. code-block:: python

    from pykeen.pipeline import pipeline

    pipeline_result = pipeline(
        model='RotatE',
        dataset='Kinships',
        result_tracker='csv',
    )

It is placed in a subdirectory of :mod:`pystow` default data directory with PyKEEN called ``logs``,
which will likely be at ``~/.data/pykeen/logs/`` on your system. The file is named based on the
current time.

Specifying a Name
-----------------
If you want to specify the name of the log file in the default directory, use the ``name`` keyword
argument like:

.. code-block:: python

    from pykeen.pipeline import pipeline

    pipeline_result = pipeline(
        model='RotatE',
        dataset='Kinships',
        result_tracker='csv',
        result_tracker_kwargs=dict(
            name='test.csv',
        ),
    )

Additional keyword arguments are passed through to the :func:`csv.writer`. This can include
a ``delimiter``, ``dialect``, ``quotechar``, etc.

.. warning:: If you specify the file name, it will overwrite the previous log file there.

The ``path`` argument can be used instead of the ``name`` to specify an absolute path to the
log file rather than using the PyStow directory.

Combining with ``tail``
-----------------------
If you know the name of a file, you can monitor it with ``tail`` and the ``-f`` flag
like in:

.. code-block::

    $ tail -f ~/data/pykeen/logs/test.csv | grep "hits_at_10"

Pipeline Example with JSON
--------------------------
The JSON writer creates a JSONL file on which each line is a valid JSON object.
Similarly to the CSV writer, the ``name`` argument can be omitted to create a time-based
file name or given to pick the default name. The ``path`` argument can still be used to specify
an absolute path.

.. code-block:: python

    from pykeen.pipeline import pipeline

    pipeline_result = pipeline(
        model='RotatE',
        dataset='Kinships',
        result_tracker='json',
        result_tracker_kwargs=dict(
            name='test.json',
        ),
    )

The same concepts can be applied in the HPO pipeline as in the previous tracker tutorials.
Additional documentation of the valid keyword arguments can be found
under :class:`pykeen.trackers.CSVResultTracker` and :class:`pykeen.trackers.JSONResultTracker`.
