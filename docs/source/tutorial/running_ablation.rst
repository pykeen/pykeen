Running an Ablation Study
=========================

In PyKEEN, we can define an ablation study in a configuration file ('file_name.json') or within our own program.

First, we show how to define an ablation study within a configuration file. In the following example, we define an
ablation study for ComplEx over the Nations dataset, and name the file 'complex_nation.json'.
In particular, we want to asses the effect of different loss functions, i.e., the binary cross entropy loss
and the margin ranking loss, functions, and the effect of explicitly modeling inverse relations.

We start by adding metadata to our configuration file. This is helpful for a later retrospection.

.. code-block:: python

    {
        "metadata": {
            "title": "Ablation Study Over Nations for ComplEx."
        }
    }



Now, we start with the actual definition of the ablation study, and define the minimal requirements, i.e.,
the dataset(s), interaction model(s), the loss function(s), training approach(es), and the optimizer(s):

.. code-block:: python

    {
        "metadata": {
            "title": "Ablation Study Over Nations for ComplEx."
        },
        "ablation": {
            "datasets": ["nations"],
            "models":   ["ComplEx"],
            "losses": ["BCEAfterSigmoidLoss", "CrossEntropyLoss"]
            "training_loops": ["lcwa"]
        }
    }



However, we also want to measure the influence of explicitly modeling inverse relations. Therefore,
we extend the ablation study accordingly:

.. code-block:: python

    {
        "metadata": {
            "title": "Ablation Study Over Nations for ComplEx."
        },
        "ablation": {
            "datasets": ["nations"],
            "models":   ["ComplEx"],
            "losses": ["BCEAfterSigmoidLoss", "CrossEntropyLoss"]
            "training_loops": ["lcwa"],
            "create_inverse_triples": [true,false]
        }
    }

For each of the components of a knowledge graph embedding model (KGEM) that requires hyper-parameters, i.e.,
interaction model, loss function and the training approach, we provide default hyper-parameter optimization (HPO)
ranges within PyKEEN. Therefore, the definition of oour ablation study would be complete at this stage. Because,
hyper-parameter ranges are dataset dependent, user can/should define their own HPO ranges. We will show later how to
do this.
To finalize, the ablation study, we recommend to define early stopping for you ablation study which can be done as
follows:

.. code-block:: python

    {
        "metadata": {
            "title": "Ablation Study Over Nations for ComplEx."
        },
        "ablation": {
            "datasets": ["nations"],
            "models":   ["ComplEx"],
            "losses": ["BCEAfterSigmoidLoss", "CrossEntropyLoss"]
            "training_loops": ["lcwa"],
            "create_inverse_triples": [true,false],
            "stopper": "early",
            "stopper_kwargs": {
                "frequency": 5,
                "patience": 20,
                "relative_delta": 0.002,
                "metric": "hits@10"
            }
        }
    }
We define the early stopper using the key 'stopper', and through 'stopper_kwargs', we provide arguments to the
early stopper. We define that the early stopper should evaluate every 5 epochs with a patience of 20 epochs on the
validation set. In order to continue training, we expect the model to obtain an improvement > 0.2% in Hits@10.