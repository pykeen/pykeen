.. _first_steps:

First Steps
===========
.. automodule:: pykeen.pipeline.api

Loading a pre-trained Model
---------------------------
Many of the previous examples ended with saving the results using the
:meth:`pykeen.pipeline.PipelineResult.save_to_directory`. One of the
artifacts written to the given directory is the ``trained_model.pkl``
file. Because all PyKEEN models inherit from :class:`torch.nn.Module`,
we use the PyTorch mechanisms for saving and loading them. This means
that you can use :func:`torch.load` to load a model like:

.. code-block:: python

    import torch

    my_pykeen_model = torch.load('trained_model.pkl')

More information on PyTorch's model persistence can be found at:
https://pytorch.org/tutorials/beginner/saving_loading_models.html.

Mapping Entity and Relation Identifiers to their Names
------------------------------------------------------
While PyKEEN internally maps entities and relations to
contiguous identifiers, it's still useful to be able to interact
with datasets, triples factories, and models using the labels
of the entities and relations.

We can map a triples factory's entities to identifiers using
:func:`TriplesFactory.entities_to_ids` like in the following
example:

.. code-block:: python

    from pykeen.datasets import Nations

    triples_factory = Nations().training

    # Get tensor of entity identifiers
    entity_ids = torch.as_tensor(triples_factory.entities_to_ids(["china", "egypt"]))

Similarly, we can map a triples factory's relations to identifiers
using :data:`TriplesFactory.relations_to_ids` like in the following
example:

.. code-block:: python

    relation_ids = torch.as_tensor(triples_factory.relations_to_ids(["independence", "embassy"]))

.. warning::

    It's important to notice that we should use a triples factory with the same mapping
    that was used to train the model - otherwise we might end up with incorrect IDs.

Using Learned Embeddings
------------------------
The embeddings learned for entities and relations are not only useful for link
prediction (see :ref:`making_predictions`), but also for other downstream machine
learning tasks like clustering, regression, and classification.

Knowledge graph embedding models can potentially have multiple entity representations and
multiple relation representations, so they are respectively stored as sequences in the
``entity_representations`` and ``relation_representations`` attributes of each model.
While the exact contents of these sequences are model-dependent, the first element of
each is usually the "primary" representation for either the entities or relations.

Typically, the values in these sequences are instances of the :class:`pykeen.nn.representation.Embedding`.
This implements a similar, but more powerful, interface to the built-in :class:`torch.nn.Embedding`
class. However, the values in these sequences can more generally be instances of any subclasses of
:class:`pykeen.nn.representation.Representation`. This allows for more powerful encoders those in GNNs
such as :class:`pykeen.models.RGCN` to be implemented and used.

The entity representations and relation representations can be accessed like this:

.. code-block:: python

    from typing import List

    import pykeen.nn
    from pykeen.pipeline import pipeline

    result = pipeline(model='TransE', dataset='UMLS')
    model = result.model

    entity_representation_modules: List['pykeen.nn.Representation'] = model.entity_representations
    relation_representation_modules: List['pykeen.nn.Representation'] = model.relation_representations

Most models, like :class:`pykeen.models.TransE`, only have one representation for entities and one
for relations. This means that the ``entity_representations`` and ``relation_representations``
lists both have a length of 1. All of the entity embeddings can be accessed like:

.. code-block:: python

    entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
    relation_embeddings: pykeen.nn.Embedding = relation_representation_modules[0]

Since all representations are subclasses of :class:`torch.nn.Module`, you need to call them like functions
to invoke the `forward()` and get the values.

.. code-block:: python

    entity_embedding_tensor: torch.FloatTensor = entity_embeddings()
    relation_embedding_tensor: torch.FloatTensor = relation_embeddings()

The `forward()` function of all :class:`pykeen.nn.representation.Representation` takes an ``indices`` parameter.
By default, it is ``None`` and returns all values. More explicitly, this looks like:

.. code-block:: python

    entity_embedding_tensor: torch.FloatTensor = entity_embeddings(indices=None)
    relation_embedding_tensor: torch.FloatTensor = relation_embeddings(indices=None)

If you'd like to only look up certain embeddings, you can use the ``indices`` parameter
and pass a :class:`torch.LongTensor` with their corresponding indices.

You might want to detach them from the GPU and convert to a :class:`numpy.ndarray` with

.. code-block:: python

    entity_embedding_tensor = model.entity_representations[0](indices=None).detach().numpy()

.. warning::

    Some old-style models (e.g., ones inheriting from :class:`pykeen.models.EntityRelationEmbeddingModel`)
    don't fully implement the ``entity_representations`` and ``relation_representations`` interface. This means
    that they might have additional embeddings stored in attributes that aren't exposed through these sequences.
    For example, :class:`pykeen.models.TransD` has a secondary entity embedding in
    :data:`pykeen.models.TransD.entity_projections`.
    Eventually, all models will be upgraded to new-style models and this won't be a problem.

Beyond the Pipeline
-------------------
While the pipeline provides a high-level interface, each aspect of the
training process is encapsulated in classes that can be more finely
tuned or subclassed. Below is an example of code that might have been
executed with one of the previous examples.

.. code-block:: python

    >>> # Get a training dataset
    >>> from pykeen.datasets import Nations
    >>> dataset = Nations()
    >>> training_triples_factory = dataset.training

    >>> # Pick a model
    >>> from pykeen.models import TransE
    >>> model = TransE(triples_factory=training_triples_factory)

    >>> # Pick an optimizer from Torch
    >>> from torch.optim import Adam
    >>> optimizer = Adam(params=model.get_grad_params())

    >>> # Pick a training approach (sLCWA or LCWA)
    >>> from pykeen.training import SLCWATrainingLoop
    >>> training_loop = SLCWATrainingLoop(
    ...     model=model,
    ...     triples_factory=training_triples_factory,
    ...     optimizer=optimizer,
    ... )

    >>> # Train like Cristiano Ronaldo
    >>> _ = training_loop.train(
    ...     triples_factory=training_triples_factory,
    ...     num_epochs=5,
    ...     batch_size=256,
    ... )

    >>> # Pick an evaluator
    >>> from pykeen.evaluation import RankBasedEvaluator
    >>> evaluator = RankBasedEvaluator()

    >>> # Get triples to test
    >>> mapped_triples = dataset.testing.mapped_triples

    >>> # Evaluate
    >>> results = evaluator.evaluate(
    ...     model=model,
    ...     mapped_triples=mapped_triples,
    ...     batch_size=1024,
    ...     additional_filter_triples=[
    ...         dataset.training.mapped_triples,
    ...         dataset.validation.mapped_triples,
    ...     ],
    ... )
    >>> # print(results)


Preview: Evaluation Loops
-------------------------
PyKEEN is currently in the transition to use torch's data-loaders for evaluation, too.
While not being active for the high-level `pipeline`, you can already use it explicitly:

.. code-block:: python

    >>> # get a dataset
    >>> from pykeen.datasets import Nations
    >>> dataset = Nations()

    >>> # Pick a model
    >>> from pykeen.models import TransE
    >>> model = TransE(triples_factory=dataset.training)

    >>> # Pick a training approach (sLCWA or LCWA)
    >>> from pykeen.training import SLCWATrainingLoop
    >>> training_loop = SLCWATrainingLoop(
    ...     model=model,
    ...     triples_factory=dataset.training,
    ... )

    >>> # Train like Cristiano Ronaldo
    >>> _ = training_loop.train(
    ...     triples_factory=training_triples_factory,
    ...     num_epochs=5,
    ...     batch_size=256,
    ...     # NEW: validation evaluation callback
    ...     callbacks="evaluation-loop",
    ...     callback_kwargs=dict(
    ...         prefix="validation",
    ...         factory=dataset.validation,
    ...     ),
    ... )

    >>> # Pick an evaluation loop (NEW)
    >>> from pykeen.evaluation import LCWAEvaluationLoop
    >>> evaluation_loop = LCWAEvaluationLoop(
    ...     model=model,
    ...     triples_factory=dataset.testing,
    ... )

    >>> # Evaluate
    >>> results = evaluation_loop.evaluate()
    >>> # print(results)


Training Callbacks
------------------
PyKEEN allows interaction with the training loop through callbacks.
One particular use case is regular evaluation (outside of an early stopper).
The following example shows how to evaluate on the training triples on every
tenth epoch

.. code-block:: python

    from pykeen.datasets import get_dataset
    from pykeen.pipeline import pipeline

    dataset = get_dataset(dataset="nations")
    result = pipeline(
        dataset=dataset,
        model="mure",
        training_kwargs=dict(
            num_epochs=100,
            callbacks="evaluation",
            callback_kwargs=dict(
                evaluation_triples=dataset.training.mapped_triples,
                tracker="console",
                prefix="training",
            ),
        ),
    )

For further information about different result trackers, take a look at the section
on :ref:`trackers`.

Next Steps
----------
The first steps tutorial taught you how to train and use a model for some of the
most common tasks. There are several other topic-specific tutorials in the section
of the documentation. You might also want to jump ahead to the :ref:`troubleshooting`
section in case you're having trouble, or look through
`questions <https://github.com/pykeen/pykeen/issues?q=is%3Aissue+is%3Aopen+label%3Aquestion>`_
and `discussions <https://github.com/pykeen/pykeen/discussions>`_ that others have posted
on GitHub.
