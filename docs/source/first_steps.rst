First Steps
===========
.. automodule:: poem.pipeline

Beyond the Pipeline
-------------------
While the pipeline provides a high-level interface, each aspect of the
training process is encapsulated in classes that can be more finely
tuned or subclassed. Below is an example of code that might have been
executed with one of the previous examples.

.. code-block:: python

    # Get a training data set
    from poem.datasets import Nations
    dataset = Nations()
    training_triples_factory = dataset.training

    # Pick a model
    from poem.models import TransE
    model = TransE(triples_factory=training_triples_factory)

    # Pick an optimizer from Torch
    from torch.optim import Adam
    optimizer = Adam(params=model.get_grad_params())

    # Pick a training assumption (OWA or CWA)
    from poem.training import OWATrainingLoop
    training_loop = OWATrainingLoop(model=model, optimizer=optimizer)

    # Train like Cristiano Ronaldo
    training_loop.train(num_epochs=5, batch_size=256)

    # Pick an evaluator
    from poem.evaluation import RankBasedEvaluator
    evaluator = RankBasedEvaluator(model)

    # Get triples to test
    mapped_triples = dataset.testing.mapped_triples

    # Evaluate
    results = evaluator.evaluate(mapped_triples, batch_size=1024)
    print(results)
