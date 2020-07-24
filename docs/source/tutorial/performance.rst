Performance Tricks
==================
PyKEEN uses a combination of techniques under the hood that ensure efficient calculations during both training and
evaluation as well as try to maximize the utilization of the hardware at hand (currently focused on single GPU usage).

Tuple broadcasting
------------------
Interaction functions are usually only given for the standard case of scoring a single triple (h, r, t). This function
is in PyKEEN implemented in the :func:`score_hrt` method of each model, e.g. :func:`pykeen.models.distmult.score_hrt`
for the DistMult model. When training with the local closed world assumption (LCWA) or evaluating a model as well as
performing link prediction tasks the goal is to score all entities/relations for a given tuple, i.e. (h, r,), (r, t) or
(h, t). In these cases a single tuple is used many times for different entities/relations.

E.g. we want to rank all entities for a single tuple (h, r) with the DistMult model for the Freebase15k237 dataset. This
dataset contains 14,505 entites, which means that we have 14,505 (h, r, t) combinations, whereas _h_ and _r_ are
constant. Looking at the interaction function of the DistMult model, we can observe that the :math:`h*r` part causes
half of the mathematical operations to calculate :math:`h*r*t`. Therefore, calculating the :math:`h*r` part only once
and reusing it spares us half of the mathematical operations for the other 14,504 remaining entities, making the
calculations roughly twice as fast in total. The speed-up might be significantly higher in cases where the broadcasted
part has a high relative complexity compared to the overall interaction function, e.g. :class:`pykeen.models.ConvE`.

To make this technique possible, PyKEEN models have to provide an explicit broadcasting function via following methods
in the model class:
 - :func:`score_t` - Scoring all possible tail entities for a given (h, r) tuple
 - :func:`score_h` - Scoring all possible head entities for a given (r, t) tuple
 - :func:`score_r` - Scoring all possible relations for a given (h, t) tuple
The PyKEEN architecture natively supports these methods and makes use of this technique wherever possible without any
additional modifications. Providing these methods is completely optional and not required when implementing new models.

Sub-batching
------------
With growing model and dataset sizes the KGEM at hand is likely to exceed the memory provided by GPUs. Especially during
training it might be desired to train using a certain batch size. When this batch size is too big for the hardware at
hand, PyKEEN allows to set a sub-batch size in the range of :math:`[1, {batch size}[`. When the sub-batch size is set,
PyKEEN automatically accumulates the gradients after each sub-batch and clears the computational graph during training.
By doing this the obtained results are identical to training without sub-batching. Note: Not that in order to guarantee
this, not all models support sub-batching, since certain components, e.g. batch normalization, require the entire batch
to be calculated in one pass to avoid altering statistics.


GPU-Filtering with index-based masking
--------------------------------------
tbd

Automated Memory Optimization
-----------------------------
tbd
