.. _bring_your_own_data:

Bring Your Own Data
===================
As an alternative to using a pre-packaged dataset, the training and testing can be set explicitly
by file path or with instances of :class:`~pykeen.triples.TriplesFactory`. Throughout this
tutorial, the paths to the training, testing, and validation sets for built-in
:class:`~pykeen.datasets.Nations` will be used as examples.

Pre-stratified Dataset
----------------------
You've got a training and testing file as 3-column TSV files, all ready to go. You're sure that there aren't
any entities or relations appearing in the testing set that don't appear in the training set. Load them in the
pipeline like this:

.. literalinclude:: ../examples/howto/bring_your_own_data.py
    :lines: 4-14

PyKEEN will take care of making sure that the entities are mapped from their labels to appropriate integer
(technically, 0-dimensional :class:`torch.LongTensor`) indexes and that the different sets of triples
share the same mapping.

This is equally applicable for the :func:`~pykeen.hpo.hpo_pipeline`, which has a similar interface to
the :func:`~pykeen.pipeline.pipeline` as in:

.. literalinclude:: ../examples/howto/bring_your_own_data.py
    :lines: 17-27

The remainder of the examples will be for :func:`~pykeen.pipeline.pipeline`, but all work exactly the same
for :func:`~pykeen.hpo.hpo_pipeline`.

If you want to add dataset-wide arguments, you can use the ``dataset_kwargs`` argument
to the :class:`~pykeen.pipeline.pipeline` to enable options like ``create_inverse_triples=True``.

.. literalinclude:: ../examples/howto/bring_your_own_data.py
    :lines: 30-37

If you want finer control over how the triples are created, for example, if they are not all coming from
TSV files, you can use the :class:`~pykeen.triples.TriplesFactory` interface.

.. literalinclude:: ../examples/howto/bring_your_own_data.py
    :lines: 40-52

.. warning::

    The instantiation of the testing factory, we used the ``entity_to_id`` and ``relation_to_id`` keyword arguments.
    This is because PyKEEN automatically assigns numeric identifiers to all entities and relations for each triples
    factory. However, we want the identifiers to be exactly the same for the testing set as the training
    set, so we just reuse it. If we didn't have the same identifiers, then the testing set would get mixed up with
    the wrong identifiers in the training set during evaluation, and we'd get nonsense results.

The ``dataset_kwargs`` argument is ignored when passing your own :class:`~pykeen.triples.TriplesFactory`, so be
sure to include the ``create_inverse_triples=True`` in the instantiation of those classes if that's your
desired behavior as in:

.. literalinclude:: ../examples/howto/bring_your_own_data.py
    :lines: 55-71

Triples factories can also be instantiated using the ``triples`` keyword argument instead of the ``path`` argument
if you already have triples loaded in a :class:`numpy.ndarray`.

Unstratified Dataset
--------------------
It's more realistic your real-world dataset is not already stratified into training and testing sets.
PyKEEN has you covered with :func:`~pykeen.triples.CoreTriplesFactory.split`, which will allow you to create
a stratified dataset.

.. literalinclude:: ../examples/howto/bring_your_own_data.py
    :lines: 74-82

By default, this is an 80/20 split. If you want to use early stopping, you'll also need a validation set, so
you should specify the splits:

.. literalinclude:: ../examples/howto/bring_your_own_data.py
    :lines: 85-96

Bring Your Own Data with Checkpoints
------------------------------------
For a tutorial on how to use your own data together with checkpoints,
see :ref:`byod_and_checkpoints_training` and :ref:`byod_and_checkpoints_manually`.
