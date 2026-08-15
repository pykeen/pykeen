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
           :lines: 6-11

2. Load the model using the version of PyKEEN you want to use. First instantiate the model, then load the state dict:

       .. literalinclude:: /examples/howto/troubleshooting.py
           :lines: 15-21

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
    :lines: 26-42

.. warning::

    Even if the state dict can be loaded, there is still a risk that the the weights are used differently. This can lead
    to a difference in model behavior. To be sure that the model is still functioning the same way, you should also
    check some model predictions and inspect *how* the model definition has changed.
