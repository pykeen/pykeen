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

.. literalinclude:: ../examples/first_steps/load_pretrained.py
    :lines: 3-5

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

.. literalinclude:: ../examples/first_steps/entity_and_relation_mapping.py
    :lines: 4-11,38-39

Similarly, we can map a triples factory's relations to identifiers
using :data:`TriplesFactory.relations_to_ids` like in the following
example:

.. literalinclude:: ../examples/first_steps/entity_and_relation_mapping.py
    :lines: 40

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

.. literalinclude:: ../examples/first_steps/using_learned_embeddings.py
    :lines: 4-14

Most models, like :class:`pykeen.models.TransE`, only have one representation for entities and one
for relations. This means that the ``entity_representations`` and ``relation_representations``
lists both have a length of 1. All of the entity embeddings can be accessed like:

.. literalinclude:: ../examples/first_steps/using_learned_embeddings.py
    :lines: 17-24

Since all representations are subclasses of :class:`torch.nn.Module`, you need to call them like functions
to invoke the `forward()` and get the values.

.. literalinclude:: ../examples/first_steps/using_learned_embeddings.py
    :lines: 28-29

The `forward()` function of all :class:`pykeen.nn.representation.Representation` takes an ``indices`` parameter.
By default, it is ``None`` and returns all values. More explicitly, this looks like:

.. literalinclude:: ../examples/first_steps/using_learned_embeddings.py
    :lines: 33-34

If you'd like to only look up certain embeddings, you can use the ``indices`` parameter
and pass a :class:`torch.LongTensor` with their corresponding indices.

.. literalinclude:: ../examples/first_steps/using_learned_embeddings.py
    :lines: 37-39

You might want to detach them from the GPU and convert to a :class:`numpy.ndarray` with

.. literalinclude:: ../examples/first_steps/using_learned_embeddings.py
    :lines: 43

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

.. literalinclude:: ../examples/first_steps/beyond_pipeline.py
    :lines: 3-


Preview: Evaluation Loops
-------------------------
PyKEEN is currently in the transition to use torch's data-loaders for evaluation, too.
While not being active for the high-level `pipeline`, you can already use it explicitly:

.. literalinclude:: ../examples/first_steps/evaluation_loop.py
    :lines: 3-


Training Callbacks
------------------
PyKEEN allows interaction with the training loop through callbacks.
One particular use case is regular evaluation (outside of an early stopper).
The following example shows how to evaluate on the training triples on every
tenth epoch

.. literalinclude:: ../examples/first_steps/callbacks.py
    :lines: 3-

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
