Using Tensorboard
=================

`Tensorboard <https://www.tensorflow.org/tensorboard/>`_ is a service for tracking experimental results during or after
training. It is part of the larger Tensorflow project but can be used independently of it.

Installing Tensorboard
----------------------

The :mod:`tensorboard` package can either be installed directly with ``pip install tensorboard`` or with PyKEEN by using
the ``tensorboard`` extra in ``pip install pykeen[tensorboard]``.

.. note::

    Tensorboard logs can created without actually installing tensorboard itself. However, if you want to view and
    interact with the data created via the tracker, it must be installed.

Starting Tensorboard
--------------------

The :mod:`tensorboard` web application can be started from the command line with

.. code-block:: shell

    $ tensorboard --logdir=~/.data/pykeen/logs/tensorboard/

where the value passed to the ``--logdir`` is location of log directory. By default, PyKEEN logs to
``~/.data/pykeen/logs/tensorboard/``, but this is configurable. The Tensorboard can then be accessed via a browser at:
http://localhost:6006/

.. note::

    It is not required for the Tensorboard process to be running while the training is happening. Indeed, it only needs
    to be started once you want to interact with and view the logs. It can be stopped at any time and the logs will
    persist in the filesystem.

Minimal Pipeline Example
------------------------

The tensorboard tracker can be used during training with the :func:`~pykeen.pipeline.pipeline` as follows:

.. literalinclude:: /examples/howto/using_tensorboard.py
    :lines: 4-10

It is placed in a subdirectory of :mod:`pystow` default data directory of PyKEEN called ``tensorboard``, which will
likely be at ``~/.data/pykeen/logs/tensorboard`` on your system. The file is named based on the current time if no
alternative is provided.

Specifying a Log Name
---------------------

If you want to specify the name of the log file in the default directory, use the ``experiment_name`` keyword argument
like:

.. literalinclude:: /examples/howto/using_tensorboard.py
    :lines: 13-20

Specifying a Custom Log Directory
---------------------------------

If you want to specify a custom directory to store the tensorboard logs, use the ``experiment_path`` keyword argument
like:

.. literalinclude:: /examples/howto/using_tensorboard.py
    :lines: 23-30

.. warning::

    Please be aware that if you re-run an experiment using the same directory, then the logs will be combined. It is
    advisable to use a unique sub-directory for each experiment to allow for easy comparison.

Minimal HPO Pipeline Example
----------------------------

Tensorboard tracking can also be used in conjunction with a HPO pipeline as follows:

.. literalinclude:: /examples/howto/using_tensorboard.py
    :lines: 33-40

This provides a way to compare directly between different trails and parameter configurations. Please not that it is
recommended to leave the experiment name as the default value here to allow for a directory to be created per trail.
