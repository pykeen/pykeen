Using Weights and Biases
========================

`Weights and Biases <http://wandb.ai/>`_ (WANDB) is a service for tracking experimental results and various artifacts
appearing whn training ML models.

After `registering <https://app.wandb.ai/login?signup=true>`_ for WANDB, do the following:

1. Create a project in WANDB, for example, with the name ``pykeen_project`` at ``https://app.wandb.ai/<your
   username>/new``
2. Install WANDB on your machine with ``pip install wandb``
3. Setup your computer for use with WANDB by using either of the following two instructions from
   https://github.com/wandb/client#running-your-script:

   1. Navigate to https://app.wandb.ai/settings, copy your API key, and set the ``WANDB_API_KEY`` environment variable
   2. Interactively run ``wandb login``

Now you can simply specify this project name when initializing a pipeline, and everything else will work automatically!

Pipeline Example
----------------

This example shows using WANDB with the :func:`~pykeen.pipeline.pipeline` function.

.. literalinclude:: /examples/howto/using_wandb.py
    :lines: 4-13

You can navigate to the created project in WANDB and observe a running experiment. Further tweaking of appearance,
charts, and other settings is described in the official `documentation <https://docs.wandb.com/>`_

You can also specify optional ``tags`` which will appear on the website instead of randomly generated labels. All
further keyword arguments are passed to :func:`wandb.init`.

.. literalinclude:: /examples/howto/using_wandb.py
    :lines: 16-24

HPO Example
-----------

This example shows using WANDB with the :func:`~pykeen.hpo.hpo_pipeline` function.

.. literalinclude:: /examples/howto/using_wandb.py
    :lines: 27-38

It's safe to specify the experiment name during HPO. Several runs will be sent to the same experiment under different
hashes. However, specifying the experiment name is advisable more for single runs and not for batches of multiple runs.

Additional documentation of the valid keyword arguments can be found under :class:`~pykeen.trackers.WANDBResultTracker`.
