Knowledge Graph Embedding (KGE) Models
======================================

Currently, 10 KGE models are available within PyKEEN, and every model has its set of hyper-parameters.
Every KGE model has a class variable called **hyper_params** (e.g. **TransE.hyper_params**) which contains all
the hyper-parameter values for the respective model. If you define an experiment programmatically you should provide
in your configuration dictionary the values for each hyper-parameter. In case you define your experiment through the
CLI, PyKEEN will ensure that a correct configuration is created.


.. automodule:: pykeen.kge_models
   :members:

