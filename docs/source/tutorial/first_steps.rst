First Steps
===========
.. automodule:: pykeen.pipeline

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

Beyond the Pipeline
-------------------
While the pipeline provides a high-level interface, each aspect of the
training process is encapsulated in classes that can be more finely
tuned or subclassed. Below is an example of code that might have been
executed with one of the previous examples.

.. code-block:: python

    # Get a training data set
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
