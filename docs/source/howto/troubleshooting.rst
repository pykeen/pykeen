.. _troubleshooting:

Troubleshooting
===============

Loading a Model from an Old Version of PyKEEN
---------------------------------------------

If your model was trained on a different version of PyKEEN, you might have difficulty loading the model using
``torch.load('trained_model.pkl')``.

This could be due to one or both of the following:

1. The model class structure might have changed.
2. The model weight names might have changed.

Note that PyKEEN currently cannot support model migration. Please attempt the following steps to load the model.

If the model class structure has changed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You will likely see an exception like this one: ``ModuleNotFoundError: No module named ...``

In this case, try to instantiate the model class directly and only load the state dict from the model file.

1. Save the model's ``state_dict`` using the version of PyKEEN used for training:

       .. literalinclude:: /examples/howto/troubleshooting.py
           :lines: 4-9

2. Load the model using the version of PyKEEN you want to use. First instantiate the model, then load the state dict:

       .. literalinclude:: /examples/howto/troubleshooting.py
           :lines: 12-18

If the model weight names have changed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You will likely see an exception similar to this one:

.. code-block::

    RuntimeError: Error(s) in loading state_dict for RotatE:
    Missing key(s) in state_dict: "entity_representations.0._embeddings.weight", "relation_representations.0._embeddings.weight".
    Unexpected key(s) in state_dict: "regularizer.weight", "regularizer.regularization_term", "entity_embeddings._embeddings.weight", "relation_embeddings._embeddings.weight".

In this case, you need to inspect the state-dict dictionaries in the different version, and try to match the keys. Then
modify the state dict accordingly before loading it. For example:

.. literalinclude:: /examples/howto/troubleshooting.py
    :lines: 21-37

.. warning::

    Even if the state dict can be loaded, there is still a risk that the the weights are used differently. This can lead
    to a difference in model behavior. To be sure that the model is still functioning the same way, you should also
    check some model predictions and inspect *how* the model definition has changed.

Customizing a Model's Representations
-------------------------------------

Most models, e.g., :class:`~pykeen.models.TransE`, are thin wrappers around :class:`~pykeen.models.ERModel`, which fill
in the interaction function as well as the entity and relation representations themselves. Passing any of those via
``model_kwargs`` thus fails with an exception like this one:

.. code-block::

    TypeError: TransE sets 'entity_representations_kwargs' itself, so it cannot be passed via model_kwargs. ...

Instead, use :class:`~pykeen.models.ERModel` directly and combine an interaction function with the representations of
your choice:

.. literalinclude:: /examples/howto/customizing_representations.py
    :start-after: # %% customize via ERModel
    :end-before: # %% subclass for HPO

Hyperparameter Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~pykeen.hpo.hpo_pipeline` samples flat parameters of the model's constructor, e.g., ``embedding_dim``, but not
entries of nested dictionaries like ``entity_representations_kwargs``. To optimize the hyperparameters of customized
representations, write a small subclass of :class:`~pykeen.models.ERModel`, which exposes the hyperparameters as
parameters of its constructor and provides their ranges in ``hpo_default``. This also allows ensuring that dimensions
which have to match, e.g., the entity and relation dimension for TransE, are always tied, and choosing between different
types of representations via a categorical parameter:

.. literalinclude:: /examples/howto/customizing_representations.py
    :start-after: # %% subclass for HPO
    :end-before: # %% custom objective

Note that :func:`~pykeen.hpo.hpo_pipeline` samples all parameters for each trial, i.e., ``hidden_dim`` is sampled even
if the plain embedding is chosen, where it is ignored. With the default, independent TPE sampler, this mostly means that
``hidden_dim`` itself is optimized less efficiently, since its ignored values also enter its model. It is more harmful
with grid search, where each ignored value repeats the same configuration, and with samplers that model the parameters
jointly, e.g., multivariate TPE or CMA-ES, for which ignored parameters add noise. Ignored parameters also show up in
the reported best parameters and parameter importances.

For such conditional search spaces, write your own :mod:`optuna` objective, which only suggests a parameter when it is
used, and still uses :func:`~pykeen.pipeline.pipeline` to train and evaluate the model. The multivariate TPE sampler
with ``group=True`` then models the different sub-spaces separately. Make sure to evaluate on the validation triples,
and keep the test triples for the final evaluation:

.. literalinclude:: /examples/howto/customizing_representations.py
    :start-after: # %% custom objective
