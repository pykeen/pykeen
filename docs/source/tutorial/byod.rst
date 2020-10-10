Bring Your Own Data
===================
As an alternative to using a pre-packaged dataset, the training and testing can be set
explicitly with instances of :class:`pykeen.triples.TriplesFactory`.

Pre-stratified Dataset
----------------------
You've got a training and testing file as 3-column TSV files, all ready to go. You're sure that there aren't
any entities or relations appearing in the testing set that don't appear in the training set. Load them in the
pipeline like this:

.. code-block:: python

    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline

    training = TriplesFactory(path=...)
    testing = TriplesFactory(
        path=...,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )

    pipeline_result = pipeline(
        training_triples_factory=training,
        testing_triples_factory=testing,
        model='TransE',
    )
    pipeline_result.save_to_directory('test_pre_stratified_transe')

Note that in the instantiation of the testing factory, we used the ``entity_to_id`` and ``relation_to_id``
keyword arguments. This is because PyKEEN automatically assigns numeric identifiers to all entities and relations
for each triples factory. However, we want the identifiers to be exactly the same for the testing set as the training
set, so we just reuse it. If we didn't have the same identifiers, then the testing set would get mixed up with
the wrong identifiers in the training set during evaluation, and we'd get nonsense results.

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

    pipeline_result = pipeline(
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

    pipeline_result = pipeline(
        training_triples_factory=training,
        testing_triples_factory=testing,
        validation_triples_factory=validation,
        model='TransE',
        stopper='early',
    )
    pipeline_result.save_to_directory('test_unstratified_stopped_transe')
