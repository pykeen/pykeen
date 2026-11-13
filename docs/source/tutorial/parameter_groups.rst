Using Different Learning Rates
==============================

When fine-tuning a knowledge graph embedding model, you may want to update pretrained entity embeddings more slowly than
newly initialized relation embeddings. PyTorch optimizers support this through parameter groups, each of which can have
its own learning rate.

This example follows :ref:`Beyond the Pipeline <beyond-the-pipeline>` and gives TransE's entity embeddings a learning
rate of ``1e-4`` and its relation embeddings a learning rate of ``1e-3``. Both representations are randomly initialized
here to keep the example self-contained; the rates demonstrate the configuration, rather than a recommended setting for
Nations.

.. literalinclude:: ../examples/first_steps/parameter_groups.py
    :lines: 3-

Run the example on CPU. Nations is bundled with PyKEEN and does not require a download. Training runs for five epochs,
then evaluates the model with the training and validation triples included in the filtering set. The ``losses`` variable
holds the training losses, ``learning_rates`` holds the two configured rates after training, and ``results`` holds the
evaluation metrics.

The optimizer and training loop must refer to the same model instance. Create the model and move it to the desired
device before constructing the optimizer. Include each trainable parameter in exactly one group. For TransE, the two
representation groups cover all trainable parameters; other models may have additional parameters or share weights, so
their groups need to be chosen accordingly.

This uses the existing training-loop API and does not require a custom optimizer or changes to ``pipeline()``.
