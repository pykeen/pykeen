Performance Tricks
==================
PyKEEN uses a combination of techniques under the hood that ensure efficient calculations during both training and
evaluation as well as try to maximize the utilization of the hardware at hand (currently focused on single GPU usage).

Entity and relation IDs
---------------------------------------------------------
When working with knowledge graphs (KG) the entities and relations contained in triples are usually stored as strings.
In knowledge graph embeddings models (KGEM) we aim at learning vector representations for them, such that the chosen
interaction function learns a useful scoring on top of them. We thus need a mapping from the string representations 
to vectors. Moreover, for computational efficiency, we would like to store all entity/relation embeddings in matrices.
Thus, the mapping process comprises two parts: Mapping strings to IDs, and using the IDs to access the embeddings 
(=row indices).

In PyKEEN, the mapping process takes place in :class:`pykeen.triples.TriplesFactory`. The triples factory maintains 
the sets of unique entity and relation labels and ensures that they are mapped to unique IDs ranging from 0 to 
`num_unique_entities` / `num_unique_relations`. The mappings are accessible via the attributes `entity_label_to_id`
/ `relation_label_to_id`. To improve the performance, the mapping process takes place only once, and the ID-based
triples are stored in a tensor `mapped_triples`.

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
 - :func:`score_h` - Scoring all possible head entities for a given (r, t) tuple
 - :func:`score_r` - Scoring all possible relations for a given (h, t) tuple
 - :func:`score_t` - Scoring all possible tail entities for a given (h, r) tuple
The PyKEEN architecture natively supports these methods and makes use of this technique wherever possible without any
additional modifications. Providing these methods is completely optional and not required when implementing new models.

Filtering with index-based masking
----------------------------------
In a standard evaluation setting of a KGEM for each triple (h, r, t) in the test/validation dataset two calculations are
performed:
 - the tuple (h, r) is combined with all possible tail entities t*
 - the tuple (r, t) is combined with all possible head entities h*
Afterwards the rank of (h, r, t) is compared to all possible (h, r, t*) as well as (h*, r, t) triples.

In the filtered setting, t* is not allowed to contain tail entities that would lead to (h, r, t*) triples already found
in the train dataset. Analogue to that, h* is not allowed to contain head entities leading to (h*, r, t) triples found
in the train dataset. This leads to the computational challenge that all new possible triples (h, r, t*) and (h*, r, t)
have to be checked against their existence in the train dataset. Considering a dataset like FB15k237, with almost 15,000
entities, each test triples leads to 30,000 possible new triples, which have to be checked against the train dataset.
After removing all possible entities found in the train dataset from h* and t*, new sets h** and t** are obtained that
allow to construct purely novel triples (h**, r, t) and (h, r, t**) not found in the train dataset.

To obtain very fast filtering PyKEEN combines the technique presented above in
:ref:`Aligned entity/relation ID and embeddings vector position` and :ref:`Tuple broadcasting` together with the
mechanism described below, which in our case has led up to 600,000 fold increase in speed for the filtered evaluation
compared to the mechanisms used in previous versions.

As a starting point, PyKEEN will always compute all possible scores also in the filtered setting. This is due to the
fact that the number of positive triples in average is very low and thus, few results have to be removed as well as the
fact that due to the technique presented in :ref:`Tuple broadcasting` any additionally scored entity has a marginally
low additional cost. Therefore, we start with the score vectors *score_t* for all possible triples (h, r, t*) and
*score_h* for all possible triples (h*, r, t).

Following, the sparse filters t' and h' are created, which state which of the entities would lead to triples found in
the train dataset. To achieve this we will rely on the technique presented in
:ref:`Aligned entity/relation ID and embeddings vector position`, i.e. all entity/relation IDs correspond to their
exact position in the respective embedding tensor.
As an example we take the tuple (h, r) from the test triple (h, r, t) and are interested in all tail entities t' that
should be removed from (h, r, t*) in order to obtain (h, r, t**).
This is achieved by performing the following steps:
 - Take r and compare it to the relations of all triples in the train dataset, leading to a boolean vector of the
size of number of triples contained in the train dataset, being true where any triple had the relation r
 - Take h and compare it to the head entities of all triples in the train dataset, leading to a boolean vector of the
size of number of triples contained in the train dataset, being true where any triple had the head entity h
 - Combine both boolean vectors, leading to a boolean vector of the size of number of triples contained in the train
dataset, being true where any triple had both the head entity h and the relation r
 - Convert the boolean vector to a non-zero index vector, stating at which indices the train dataset contains triples
that contain both the head entity h and the relation r, having the size of the number of non-zero elements
 - The index vector is now applied on the tail entity column of the train dataset, returning all tail entity IDs t' that
combined with h and r lead to triples contained in the train dataset
 - Finally, the t' tail entity ID index vector is applied on the initially mentioned *score_t* vector for all possible
triples (h, r, t*) and all affected scores are set to float('nan') following the IEEE-754 specification, which makes
these scores non-comparable, effectively leading to the score vector for all possible novel triples (h, r, t**).

In an analogue fashion h' is obtained and filtered from (h*, r, t) to obtain (h**, r, t).

Sub-batching
------------
With growing model and dataset sizes the KGEM at hand is likely to exceed the memory provided by GPUs. Especially during
training it might be desired to train using a certain batch size. When this batch size is too big for the hardware at
hand, PyKEEN allows to set a sub-batch size in the range of :math:`[1, {batch size}[`. When the sub-batch size is set,
PyKEEN automatically accumulates the gradients after each sub-batch and clears the computational graph during training.
This allows to train KGEM on GPU that otherwise would be too big for the hardware at hand, while the obtained results
are identical to training without sub-batching. Note: In order to guarantee this, not all models support sub-batching,
since certain components, e.g. batch normalization, require the entire batch to be calculated in one pass to avoid
altering statistics.

Automated Memory Optimization
-----------------------------
Allowing high computational throughput while ensuring that the available hardware memory is not exceeded during training
and evaluation requires the knowledge of the maximum possible training and evaluation batch size for the current model
configuration. However, determining the training and evaluation batch sizes is a tedious process, and not feasible when
a large set of heterogeneous experiments are run. Therefore, PyKEEN has an automatic memory optimization step that
computes the maximum possible training and evaluation batch sizes for the current model configuration and available
hardware before the actual calculation starts. If the user-provided batch size is too large for the used hardware, the
automatic memory optimization determines the maximum sub-batch size for training and accumulates the gradients with the
above described process :ref:`Sub-batching`. The batch sizes are determined using binary search taking into
consideration the CUDA architecture,
`<https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9926-tensor-core-performance-the-ultimate-guide.pdf>`
which ensures that the chosen batch size is the most CUDA efficient one.
