Using Checkpoints
=================
Training may take days to weeks in extreme cases when using models with many parameters or big datasets. This introduces
a large array of possible errors, e.g. session timeouts, server restarts etc., which would lead to a complete loss of
all progress made so far. To avoid this the :class:`pykeen.training.TrainingLoop` supports built-in check-points that
allow a straight-forward saving of the current training loop state and resumption of a saved
state from saved checkpoints.

How to do it
------------
To show how checkpoints are used with PyKEEN let's look at a simple example of how a model is setup.
For fixing possible errors and safety fallbacks please also look at :ref:`word_of_caution`.

.. code-block:: python

    from pykeen.models import TransE
    from pykeen.training import SLCWATrainingLoop
    from pykeen.triples import TriplesFactory
    from torch.optim import Adam

    triples_factory = Nations().training
    model = TransE(
        triples_factory=triples_factory,
        random_seed=123,
    )

    optimizer = Adam(params=model.get_grad_params())
    training_loop = SLCWATrainingLoop(model=model, optimizer=optimizer)

At this point we have a model, dataset and optimizer all setup in a training loop and are ready to train the model with
the ``training_loop``'s method :func:`pykeen.training.TrainingLoop.train`. To enable checkpoints all you have to do is
setting the function argument ``checkpoint_file`` to the name you would like it to have.
Optionally, you can set the path to where you want the checkpoints to be saved by setting the ``checkpoint_root``
argument with a string or a :class:`pathlib.Path` object containing your desired root path. If you didn't set the
``checkpoint_root`` argument, your checkpoints will be saved in the ``PYKEEN_HOME`` directory that is defined in
:mod:`pykeen.constants`, which is a subdirectory in your home directory, e.g. ``~/.pykeen/checkpoints``.
Furthermore, you can set the checkpoint frequency, i.e. how often checkpoints should be saved given in minutes, by
setting the argument ``checkpoint_frequency`` with an integer. The default frequency is 30 minutes and setting it to
``0`` will cause the training loop to save a checkpoint after each epoch.

Here is an example:

.. code-block:: python

    losses = training_loop.train(
        num_epochs=1000,
        checkpoint_file='my_checkpoint.pt',
        checkpoint_frequency=5,
    )

With this code we have started the training loop with the above defined KGEM. The training loop will save a checkpoint
in the ``my_checkpoint.pt`` file, which will be saved in the ``~/.pykeen/checkpoints/`` directory, since we haven't
set the argument ``checkpoint_root``.
The checkpoint file will be saved after 5 minutes since starting the training loop or the last time a checkpoint was
saved and the epoch finishes, i.e. when one epoch takes 10 minutes the checkpoint will be saved after 10 minutes.
In addition, checkpoints are always saved when the early stopper stops the training loop or the last epoch was finished.

Let's assume you were anticipative, saved checkpoints and your training loop crashed after 200 epochs.
Now you would like to resume from the last checkpoint. All you have to do is to rerun the **exact same code** as above
and PyKEEN will smoothly start from the given checkpoint. Since PyKEEN stores all random states as well as the
states of the model, optimizer and early stopper, the results will be exactly the same compared to running the
training loop uninterruptedly. Of course, PyKEEN will also continue saving new checkpoints even when
resuming from a previous checkpoint.

On top of resuming interrupted training loops you can also resume training loops that finished successfully.
E.g. the above training loop finished successfully after 1000 epochs, but you would like to
train the same model from that state for 2000 epochs. All you have have to do is to change the argument
``num_epochs`` in the above code to:

.. code-block:: python

    losses = training_loop.train(
        num_epochs=2000,
        checkpoint_file='my_checkpoint.pt',
        checkpoint_frequency=5,
    )

and now the training loop will resume from the state at 1000 epochs and continue to train until 2000 epochs.

Another nice feature is that the checkpoints functionality integrates with the pipeline. This means that you can simply
define a pipeline like this:

.. code-block:: python

    from pykeen.pipeline import pipeline
    pipeline_result = pipeline(
        dataset='Nations',
        model='TransE',
        optimizer='Adam',
        training_kwargs=dict(num_epochs=1000, checkpoint_file='my_checkpoint.pt', checkpoint_frequency=5),
    )

Again, assuming that e.g. this pipeline crashes after 200 epochs, you can simply execute **the same code** and the
pipeline will load the last state from the checkpoint file and continue training as if nothing happened.

.. todo:: Tutorial on recovery from hpo_pipeline.

Checkpoints on Failure
----------------------
In cases where you only would like to save checkpoints whenever the training loop might fail, you can use the argument
``checkpoint_on_failure=True``, like:

.. code-block:: python

    losses = training_loop.train(
        num_epochs=2000,
        checkpoint_on_failure=True,
    )

This option differs from ordinary checkpoints, since ordinary checkpoints are only saved
after a successful epoch. When saving checkpoints due to failure of the training loop there is no guarantee that all
random states can be recovered correctly, which might cause problems with regards to the reproducibility of that
specific training loop. Therefore, these checkpoints are saved with a distinct checkpoint name, which will be
``PyKEEN_just_saved_my_day_{datetime}.pt`` in the given checkpoint_root, even when you also opted to use ordinary
checkpoints as defined above, e.g. with this code:

.. code-block:: python

    losses = training_loop.train(
        num_epochs=2000,
        checkpoint_file='my_checkpoint.pt',
        checkpoint_frequency=5,
        checkpoint_on_failure=True,
    )

Note: Use this argument with caution, since every failed training loop will create a distinct checkpoint file.

.. _word_of_caution:

Word of Caution and Possible Errors
-----------------------------------
When using checkpoints and trying out several configurations, which in return result in multiple different checkpoints,
the inherent risk of overwriting checkpoints arises. This would naturally happen when you change the configuration of
the KGEM, but don't change the ``checkpoint_file`` argument.
To prevent this from happening, PyKEEN makes a hash-sum comparison of the configurations of the checkpoint and
the one of the current configuration at hand. When these don't match, PyKEEN won't accept the checkpoint and raise
an error.

In case you want to overwrite the previous checkpoint file with a new configuration, you have to delete it explicitly.
The reason for this behavior is three-fold:

1. This allows a very easy and user friendly way of resuming an interrupted training loop by simply re-running
   the exact same code.
2. By explicitly requiring to name the checkpoint files the user controls the naming of the files and thus makes
   it easier to keep an overview.
3. Creating new checkpoint files for each run will lead most users to inadvertently spam their file systems with
   unused checkpoints that with ease can add up to hundred of GBs when running many experiments.
