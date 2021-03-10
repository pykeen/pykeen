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

Using Learned Embeddings
------------------------
The embeddings learned for entities and relations are useful for link
prediction in PyKEEN, but also generally useful for other downstream
machine learning tasks like clustering, regression, and classification.

The embeddings themselves are typically stored in a :class:`pykeen.nn.Embedding`,
which wraps the :class:`torch.nn.Embedding` with several key values. They
can be accessed like this:

.. code-block:: python

    from pykeen.pipeline import pipeline
    result = pipeline(model='TransE', dataset='UMLS')
    model = result.model
    entity_embeddings = model.entity_embeddings._embeddings.weight.data
    relation_embeddings = model.relation_embeddings._embeddings.weight.data

However, the :class:`pykeen.nn.Embedding` inherits from the more generalizable
:class:`pykeen.nn.RepresentationModule`, which can be used for alternative
implementations. The new-style way to access the embeddings is now like this:

.. code-block:: python

    entity_embeddings = model.entity_embeddings()

More explicitly:

.. code-block:: python

    entity_embeddings = model.entity_embeddings(indices=None)

If you'd like to only look up certain embeddings, you can use the ``indices`` parameter
and pass a :class:`torch.LongTensor` with their corresponding indices.

New-style models (e.g., ones inheriting from :class:`pykeen.models.ERModel`) are
generalized to allow for multiple entity representations and
relation representations. This means that for some models, you should access them with:

.. code-block:: python

    entity_embeddings = model.entity_representations[0]()

Where ``[0]`` corresponds to the ordering of representations defined in
the interaction function. Some models may provide Pythonic properties that
provide a vanity attribute to the instance of the class for a specific
entity or relation representation.

Beyond the Pipeline
-------------------
While the pipeline provides a high-level interface, each aspect of the
training process is encapsulated in classes that can be more finely
tuned or subclassed. Below is an example of code that might have been
executed with one of the previous examples.

.. code-block:: python

    # Get a training dataset
    from pykeen.datasets import Nations
    dataset = Nations()
    training_triples_factory = dataset.training

    # Pick a model
    from pykeen.models import TransE
    model = TransE(triples_factory=training_triples_factory)

    # Pick an optimizer from Torch
    from torch.optim import Adam
    optimizer = Adam(params=model.get_grad_params())

    # Pick a training approach (sLCWA or LCWA)
    from pykeen.training import SLCWATrainingLoop
    training_loop = SLCWATrainingLoop(model=model, optimizer=optimizer)

    # Train like Cristiano Ronaldo
    training_loop.train(num_epochs=5, batch_size=256)

    # Pick an evaluator
    from pykeen.evaluation import RankBasedEvaluator
    evaluator = RankBasedEvaluator()

    # Get triples to test
    mapped_triples = dataset.testing.mapped_triples

    # Evaluate
    results = evaluator.evaluate(model, mapped_triples, batch_size=1024)
    print(results)
