Running an Ablation Study
=========================
You want to find out which loss function and training approach is best-suited for your interaction model
(model architecture)? Then performing an ablation study is the way to go!

In general, an ablation study is a set of experiments in which components of a machine learning system are removed/replaced in order
to measure the impact of these components on the system's performance. In the context of knowledge graph embedding
models, typical ablation studies involve investigating different loss functions, training approaches, negative
samplers, and the explicit modeling of inverse relations. For a specific model composition based on these components,
the best set of hyper-parameter values, e.g., embedding dimension, learning rate, batch size, loss function-specific
hyper-parameters such as the margin value in the margin ranking loss need to be determined. This is accomplished by
a process called hyper-parameter optimization. Different approaches are have been proposed, of which random search and
grid search are very popular.


In PyKEEN, we can define an ablation study in a configuration file (``file_name.json``) or within our own program.

First, we show how to define an ablation study within a configuration file. In the following example, we define an
ablation study for ComplEx over the Nations dataset, and name the file ``complex_nation.json``.
In particular, we want to asses the effect of different loss functions, i.e., the binary cross entropy loss
and the margin ranking loss, and the effect of explicitly modeling inverse relations.

We start by adding metadata to our configuration file. This is helpful for a later retrospection.

.. code-block:: javascript

    {
        "metadata": {
            "title": "Ablation Study Over Nations for ComplEx."
        }
    }

Now, we start with the actual definition of the ablation study, and define the minimal requirements, i.e.,
the dataset(s), interaction model(s), the loss function(s), training approach(es), and the optimizer(s):

.. code-block:: javascript

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

.. code-block:: javascript

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

.. code-block:: javascript

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

We define the early stopper using the key ``stopper``, and through ``stopper_kwargs``, we provide arguments to the
early stopper. We define that the early stopper should evaluate every 5 epochs with a patience of 20 epochs on the
validation set. In order to continue training, we expect the model to obtain an improvement > 0.2% in Hits@10.

After defining the ablation study, we need to define the HPO settings for each experiment within our ablation
study. Remember that for each ablation-experiment we perform an HPO in order to determine the best hyper-parameters
for the currently investigated model. In PyKEEN, we use
`Optuna <https://github.com/optuna/optunahttps://github.com/optuna/optuna>`_  as HPO framework. Therefore, we define
the arguments required by Optuna in our configuration:

.. code-block:: javascript

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
            },
        "optuna": {
            "n_trials": 2,
            "timeout": 300,
            "metric": "hits@10",
            "direction": "maximize",
            "sampler": "random",
            "pruner": "nop"
            }
        }
    }


The dictionary ``optuna`` contains all Optuna related arguments. Within this dictionary, we set the number
of HPO iterations for each experiment to 2 using the argument ``n_trials``, set a ``timeout`` of 300 seconds
(the HPO will be terminated after ``n_trials`` or ``timeout`` seconds depending on what occurs first), the ``metric`` to
optimize, define whether the metric should be maximized or minimized using the key ``direction``, define random search
as HPO algorithm using the key ``sampler``, and finally define that we do not use a pruner for pruning unpromising trials
(note that we use early stopping instead).
Now that our configuration is complete, we can start the ablation study using the CLI-function
:func:`pykeen.experiments.cli.ablation`:

>>> pykeen experiments ablation path/to/complex_nation.json -d path/to/output/directory


To measure the variance in performance, we can additionally define how often we want to re-train and re-evaluate
the best model of each ablation-experiment using the option `-r`/`--best-replicates`:

>>> pykeen experiments ablation path/to/complex_nation.json -d path/to/output/directory -r 5

Eager to check out the results? Then navigate to the output directory ``path/to/output/directory`` in which you will
find a directory whose name contains a timestamp and a unique id. Within this directory, you will find subdirectories,
e.g., ``0000_nations_complex`` which contains all experimental artifacts of one specific ablation experiment of the
defined ablation study. The most relevant subdirectory is ``best_pipeline`` which comprises the artifacts of the best
performing experiment, including it's definition in ``pipeline_config.json``,  the obtained results, and the trained
model(s) in the sub-directory ``replicates``. The number of replicates in ``replicates`` corresponds to the number
provided with the argument ``-r``.
Additionally, you are provided with further information about the ablation study in the root directory: ``study.json``
describes the ablation experiment, ``hpo_config.json`` describes the HPO of the ablation experiment, ``trials.tsv``
provides an overview of each HPO-experiment.

Define Your Own HPO Ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned above, we provide default hyper-parameters/hyper-parameter ranges for each
hyper-parameter.
However, these default values/ranges don't ensure to optimally solve your problem. Therefore,
it is time that you define your own ranges, and we show you how to do it!
To accomplish this, two dictionaries are essential, ``kwargs`` that is used to assign the hyper-parameters fixed
values, and ``kwargs_ranges`` to define ranges of values from which to sample from.

Let's start with assigning HPO ranges to hyper-parameters belonging to the interaction model. This can be achieved
by using the dictionary ``model_to_model_kwargs_ranges``:

.. code-block:: javascript
    {
        ...

        "ablation":{
            ...
            "model_to_model_kwargs_ranges":{
                "ComplEx": {
                    "embedding_dim": {
                        "type": "int",
                        "low": 4,
                        "high": 6,
                        "scale": "power_two"
                    }
                }
            }
        }
    }

We defined an HPO range for the embedding dimension. Since the ``scale`` is ``power_two``, the lower bound (``low``) 4,
the upper bound ``high`` 6, the embedding dimension is sampled from the set :math:`\{2^4,2^5, 2^6\}`.

Next, we fix the number of training epochs to 500 using the key ``model_to_trainer_to_training_kwargs`` and define
a range for the batch size using ``model_to_trainer_to_training_kwargs_ranges`` since these are hyper-parameters of the
training function:

.. code-block:: javascript
    {
        ...

        "ablation":{
            ...
            "model_to_model_kwargs_ranges":{
                "ComplEx": {
                    "embedding_dim": {
                        "type": "int",
                        "low": 4,
                        "high": 6,
                        "scale": "power_two"
                    }
                }
            },
            "model_to_trainer_to_training_kwargs": {
                "ComplEx": {
                    "lcwa": {
                        "num_epochs": 500
                    }
                }
            },
            "model_to_trainer_to_training_kwargs_ranges": {
                "ComplEx": {
                    "lcwa": {
                        "label_smoothing": {
                            "type": "float",
                            "low": 0.001,
                            "high": 1.0,
                            "scale": "log"
                        },
                        "batch_size": {
                            "type": "int",
                            "low": 7,
                            "high": 9,
                            "scale": "power_two"
                        }
                    }
                }
            }
        }
    }
Finally, we define a range for the learning rate which is a hyper-parameter of the optimizer:

.. code-block:: javascript
    {
        ...

        "ablation":{
            ...
            "model_to_model_kwargs_ranges":{
                "ComplEx": {
                    "embedding_dim": {
                        "type": "int",
                        "low": 4,
                        "high": 6,
                        "scale": "power_two"
                    }
                }
            },
            "model_to_trainer_to_training_kwargs": {
                "ComplEx": {
                    "lcwa": {
                        "num_epochs": 500
                    }
                }
            },
            "model_to_trainer_to_training_kwargs_ranges": {
                "ComplEx": {
                    "lcwa": {
                        "label_smoothing": {
                            "type": "float",
                            "low": 0.001,
                            "high": 1.0,
                            "scale": "log"
                        },
                        "batch_size": {
                            "type": "int",
                            "low": 7,
                            "high": 9,
                            "scale": "power_two"
                        }
                    }
                }
            },
            "model_to_optimizer_to_optimizer_kwargs_ranges": {
                "ComplEx": {
                    "adam": {
                        "lr": {
                            "type": "float",
                            "low": 0.001,
                            "high": 0.1,
                            "scale": "log"
                        }
                    }
                }
            }
        }
    }
We decided to use Adam as an optimizer, and we defined a ``log`` ``scale`` for the learning rate, i.e., the learning
rate is sampled from the interval :math:`[0.001, 0.1)`.

Now that we defined our own hyper-parameter values and ranges, let's have a look at the overall configuration:

.. code-block:: javascript

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
            },
            "model_to_model_kwargs_ranges":{
                "ComplEx": {
                    "embedding_dim": {
                        "type": "int",
                        "low": 4,
                        "high": 6,
                        "scale": "power_two"
                    }
                }
            },
            "model_to_trainer_to_training_kwargs": {
                "ComplEx": {
                    "lcwa": {
                        "num_epochs": 500
                    }
                }
            },
            "model_to_trainer_to_training_kwargs_ranges": {
                "ComplEx": {
                    "lcwa": {
                        "label_smoothing": {
                            "type": "float",
                            "low": 0.001,
                            "high": 1.0,
                            "scale": "log"
                        },
                        "batch_size": {
                            "type": "int",
                            "low": 7,
                            "high": 9,
                            "scale": "power_two"
                        }
                    }
                }
            },
            "model_to_optimizer_to_optimizer_kwargs_ranges": {
                "ComplEx": {
                    "adam": {
                        "lr": {
                            "type": "float",
                            "low": 0.001,
                            "high": 0.1,
                            "scale": "log"
                        }
                    }
                }
            }
        "optuna": {
            "n_trials": 2,
            "timeout": 300,
            "metric": "hits@10",
            "direction": "maximize",
            "sampler": "random",
            "pruner": "nop"
            }
        }
    }

We are expected to provide the configuration for the keys ``datasets``, ``models``, ``losses``, ``optimizers``, and
``training_loops``. For all other components and hype-parameters, PyKEEN will provide default values/ranges.
However, for achieving optimal performance, we should carefully define the hyper-parameters ourselves, as explained
above. Note that there many more ranges to configure such hyper-parameters for the loss functions, or the negative
samplers. Check out the examples provided in `tests/resources/hpo_complex_nations.json`` how to define the
ranges for other components.

Run an Ablation Study With Your Own Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We showed to run an ablation study with PyKEEN integrated dataset. Now you are asking yourself, whether you can
run ablations studies with your own data? Yes, you can!
It requires a minimal change compared to the previous configuration:

.. code-block:: javascript

    {   ...
        "ablation": {
            "datasets": [
                {
                    "training": "/path/to/your/train.txt",
                    "validation": "/path/to/your/validation.txt",
                    "testing": "/path/to/your/test.txt"
                }
            ],
        }
        ...
    }

In the dataset field, you don't provide a list of dataset names but dictionaries containing the paths
to your train-validation-test splits. Check out ``tests/resources/hpo_complex_your_own_data.json`` for a
concrete example. Yes, that's all.