Bring Your Own Data
===================
As an alternative to using a pre-packaged dataset, the training and testing can be set explicitly
by file path or with instances of :class:`pykeen.triples.TriplesFactory`.

Pre-stratified Dataset
----------------------
You've got a training and testing file as 3-column TSV files, all ready to go. You're sure that there aren't
any entities or relations appearing in the testing set that don't appear in the training set. Load them in the
pipeline like this:

.. code-block:: python

    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline

    training_path: str = ...
    testing_path: str = ...

    result = pipeline(
        training_triples_factory=training_path,
        testing_triples_factory=testing_path,
        model='TransE',
    )
    result.save_to_directory('test_pre_stratified_transe')

PyKEEN will take care of making sure that the entities are mapped from their labels to appropriate integer
(technically, 0-dimensional :class:`torch.LongTensor`) indexes and that the different sets of triples
share the same mapping.

This is equally applicable for the :func:`pykeen.hpo.hpo_pipeline`, which has a similar interface to
the :func:`pykeen.pipeline.pipeline` as in:

.. code-block:: python

    from pykeen.triples import TriplesFactory
    from pykeen.hpo import hpo_pipeline

    training_path: str = ...
    testing_path: str = ...

    result = hpo_pipeline(
        n_trials=30,
        training_triples_factory=training_path,
        testing_triples_factory=testing_path,
        model='TransE',
    )
    result.save_to_directory('test_hpo_pre_stratified_transe')

The remainder of the examples will be for :func:`pykeen.pipeline.pipeline`, but all work exactly the same
for :func:`pykeen.hpo.hpo_pipeline`.

If you want to add dataset-wide arguments, you can use the ``dataset_kwargs`` argument
to the :class:`pykeen.pipeline.pipeline` to enable options like ``create_inverse_triples=True``.

.. code-block:: python

    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline

    training_path: str = ...
    testing_path: str = ...

    result = pipeline(
        training_triples_factory=training_path,
        testing_triples_factory=testing_path,
        dataset_kwargs={'create_inverse_triples': True},
        model='TransE',
    )
    result.save_to_directory('test_pre_stratified_transe')

If you want finer control over how the triples are created, for example, if they are not all coming from
TSV files, you can use the :class:`pykeen.triples.TriplesFactory` interface.

.. code-block:: python

    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline

    training_path: str = ...
    testing_path: str = ...

    training = TriplesFactory(path=training_path)
    testing = TriplesFactory(
        path=testing_path,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )

    result = pipeline(
        training_triples_factory=training,
        testing_triples_factory=testing,
        model='TransE',
    )
    pipeline_result.save_to_directory('test_pre_stratified_transe')

.. warning::

    The instantiation of the testing factory, we used the ``entity_to_id`` and ``relation_to_id`` keyword arguments.
    This is because PyKEEN automatically assigns numeric identifiers to all entities and relations for each triples
    factory. However, we want the identifiers to be exactly the same for the testing set as the training
    set, so we just reuse it. If we didn't have the same identifiers, then the testing set would get mixed up with
    the wrong identifiers in the training set during evaluation, and we'd get nonsense results.

The ``dataset_kwargs`` argument is ignored when passing your own :class:`pykeen.triples.TriplesFactory`, so be
sure to include the ``create_inverse_triples=True`` in the instantiation of those classes if that's your
desired behavior as in:

.. code-block:: python

    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline

    training_path: str = ...
    testing_path: str = ...

    training = TriplesFactory(
        path=training_path,
        create_inverse_triples=True,
    )
    testing = TriplesFactory(
        path=testing_path,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
        create_inverse_triples=True,
    )

    result = pipeline(
        training_triples_factory=training,
        testing_triples_factory=testing,
        model='TransE',
    )
    result.save_to_directory('test_pre_stratified_transe')

Triples factories can also be instantiated using the ``triples`` keyword argument instead of the ``path`` argument
if you already have triples loaded in a :class:`numpy.ndarray`.

Unstratified Dataset
--------------------
It's more realistic your real-world dataset is not already stratified into training and testing sets.
PyKEEN has you covered with :func:`pykeen.triples.TriplesFactory.split`, which will allow you to create
a stratified dataset.

.. code-block:: python

    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline

    tf = TriplesFactory(path=...)
    training, testing = tf.split()

    result = pipeline(
        training_triples_factory=training,
        testing_triples_factory=testing,
        model='TransE',
    )
    pipeline_result.save_to_directory('test_unstratified_transe')

By default, this is an 80/20 split. If you want to use early stopping, you'll also need a validation set, so
you should specify the splits:

.. code-block:: python

    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline

    tf = TriplesFactory(path=...)
    training, testing, validation = tf.split([.8, .1, .1])

    result = pipeline(
        training_triples_factory=training,
        testing_triples_factory=testing,
        validation_triples_factory=validation,
        model='TransE',
        stopper='early',
    )
    pipeline_result.save_to_directory('test_unstratified_stopped_transe')
