Performance Tricks
==================
PyKEEN uses a combination of techniques to promote efficient calculations during training/evaluation
and tries to maximize the utilization of the available hardware (currently focused on single GPU usage).

.. _entity_and_relation_ids:

Entity and Relation IDs
-----------------------
Entities and relations in triples are usually stored as strings.
Because KGEMs aim at learning vector representations for these entities and relations such that the chosen
interaction function learns a useful scoring on top of them, we need a mapping from the string representations
to vectors. Moreover, for computational efficiency, we would like to store all entity/relation embeddings in matrices.
Thus, the mapping process comprises two parts: Mapping strings to IDs, and using the IDs to access the embeddings
(=row indices).

In PyKEEN, the mapping process takes place in :class:`pykeen.triples.TriplesFactory`. The triples factory maintains
the sets of unique entity and relation labels and ensures that they are mapped to unique integer IDs on
$[0,\text{num_unique_entities})$ for entities and $[0, \text{num_unique_relations})$. The mappings are respectively
accessible via the attributes :data:``pykeen.triples.TriplesFactory.entity_label_to_id`` and
:data:``pykeen.triples.TriplesFactory.relation_label_to_id``.

To improve the performance, the mapping process takes place only once, and the ID-based
triples are stored in a tensor :data:``pykeen.triples.TriplesFactory.mapped_triples``.

.. _tuple_broadcasting:

Tuple Broadcasting
------------------
Interaction functions are usually only given for the standard case of scoring a single triple $(h, r, t)$. This function
is in PyKEEN implemented in the :func:`pykeen.models.base.Model.score_hrt` method of each model, e.g.
:func:`pykeen.models.DistMult.score_hrt` for :class:`pykeen.models.DistMult`. When training under the local closed
world assumption (LCWA), evaluating a model, and performing the link prediction task, the goal is to score all
entities/relations for a given tuple, i.e. $(h, r)$, $(r, t)$ or $(h, t)$. In these cases a single tuple is used
many times for different entities/relations.

For example, we want to rank all entities for a single tuple $(h, r)$ with :class:`pykeen.models.DistMult` for the
:class:`pykeen.datasets.FB15k237`. This dataset contains 14,505 entities, which means that there are 14,505 $(h, r, t)$
combinations, whereas $h$ and $r$ are constant. Looking at the interaction function of :class:`pykeen.models.DistMult`,
we can observe that the :math:`h \odot r` part causes half of the mathematical operations to calculate
:math:`h \odot r \odot t`. Therefore, calculating the :math:`h \odot r` part only once and reusing it spares us
half of the mathematical operations for the other 14,504 remaining entities, making the calculations roughly
twice as fast in total. The speed-up might be significantly higher in cases where the broadcasted part has a high
relative complexity compared to the overall interaction function, e.g. :class:`pykeen.models.ConvE`.

To make this technique possible, PyKEEN models have to provide an explicit broadcasting function via following methods
in the model class:

 - :func:`pykeen.models.base.Model.score_h` - Scoring all possible head entities for a given $(r, t)$ tuple
 - :func:`pykeen.models.base.Model.score_r` - Scoring all possible relations for a given $(h, t)$ tuple
 - :func:`pykeen.models.base.Model.score_t` - Scoring all possible tail entities for a given $(h, r)$ tuple

The PyKEEN architecture natively supports these methods and makes use of this technique wherever possible without any
additional modifications. Providing these methods is completely optional and not required when implementing new models.

Filtering with Index-based Masking
----------------------------------
In this example, it is given a knowledge graph $\mathcal{K} \subseteq \mathcal{E} \times \mathcal{R} \times \mathcal{E}$
and disjoint unions of $\mathcal{K}$ in training triples $\mathcal{K}_{train}$, testing triples $\mathcal{K}_{test}$,
and validation triples $\mathcal{K}_{val}$. The same operations are performed on $\mathcal{K}_{test}$ and
$\mathcal{K}_{val}$, but only $\mathcal{K}_{test}$ will be given as example in this section.

Two calculations are performed for each test triple $(h, r, t) \in \mathcal{K}_{test}$ during standard evaluation of
a knowledge graph embedding model with interaction function
$f:\mathcal{E} \times \mathcal{R} \times \mathcal{E} \rightarrow \mathbb{R}$ for the link prediction task:

1. $(h, r)$ is combined with all possible tail entities $t' \in \mathcal{E}$ to make triples
   $T_{h,r} = \{(h,r,t') \mid t' \in \mathcal{E}\}$
2. $(r, t)$ is combined with all possible head entities $h' \in \mathcal{E}$ to make triples
   $H_{r,t} = \{(h',r,t) \mid h' \in \mathcal{E}\}$

Finally, the ranking of $(h, r, t)$ is calculated against all $(h, r, t') \in T_{h,r}$
and $(h', r, t) \in H_{r,t}$ triples with respect to the interaction function $f$.

In the filtered setting, $T_{h,r}$ is not allowed to contain tail entities $(h, r, t') \in \mathcal{K}_{train}$
and $H_{r,t}$ is not allowed to contain head entities leading to $(h', r, t) \in \mathcal{K}_{train}$ triples
found in the train dataset. Therefore, their definitions could be amended like:

- $T^{\text{filtered}}_{h,r} = \{(h,r,t') \mid t' \in \mathcal{E}\} \setminus \mathcal{K}_{train}$
- $H^{\text{filtered}}_{r,t} = \{(h',r,t) \mid h' \in \mathcal{E}\} \setminus \mathcal{K}_{train}$

While this easily defined theoretically, it poses several practical challenges.
For example, it leads to the computational challenge that all new possible triples $(h, r, t') \in T_{h,r}$ and
$(h', r, t) \in H_{r,t}$ must be enumerated and then checked for existence in $\mathcal{K}_{train}$.
Considering a dataset like :class:`pykeen.datasets.FB15k237` that has almost 15,000 entities, each test triple
$(h,r,t) \in \mathcal{K}_{test}$ leads to $2 * | \mathcal{E} | = 30,000$ possible new triples, which have to be
checked against the train dataset and then removed.

To obtain very fast filtering, PyKEEN combines the technique presented above in
:ref:`entity_and_relation_ids` and :ref:`tuple_broadcasting` together with the following
mechanism, which in our case has led to a 600,000 fold increase in speed for the filtered evaluation
compared to the mechanisms used in previous versions.

As a starting point, PyKEEN will always compute scores for all triples in $H_{r,t}$ and $T_{h,r}$, even in
the filtered setting. Because the number of positive triples on average is very low, few results have to be removed.
Additionally, due to the technique presented in :ref:`tuple_broadcasting`, scoring extra entities has a
marginally low cost. Therefore, we start with the score vectors from :func:`pykeen.models.base.Model.score_t`
for all triples $(h, r, t') \in H_{r,t}$ and from :func:`pykeen.models.base.Model.score_h`
for all triples $(h', r, t) \in T_{h,r}$.

Following, the sparse filters $\mathbf{f}_t \in \mathbb{B}^{| \mathcal{E}|}$ and
$\mathbf{f}_h \in \mathbb{B}^{| \mathcal{E}|}$ are created, which state which of the entities would lead to triples
found in the train dataset. To achieve this we will rely on the technique presented in
:ref:`entity_and_relation_ids`, i.e. all entity/relation IDs correspond to their
exact position in the respective embedding tensor.
As an example we take the tuple $(h, r)$ from the test triple $(h, r, t) \in \mathcal{K}_{test}$ and are interested
in all tail entities $t'$ that should be removed from $T_{h,r}$ in order to obtain $T^{\text{filtered}}_{h,r}$.
This is achieved by performing the following steps:

1. Take $r$ and compare it to the relations of all triples in the train dataset, leading to a boolean vector of the
   size of number of triples contained in the train dataset, being true where any triple had the relation $r$
2. Take $h$ and compare it to the head entities of all triples in the train dataset, leading to a boolean vector of the
   size of number of triples contained in the train dataset, being true where any triple had the head entity $h$
3. Combine both boolean vectors, leading to a boolean vector of the size of number of triples contained in the train
   dataset, being true where any triple had both the head entity $h$ and the relation $r$
4. Convert the boolean vector to a non-zero index vector, stating at which indices the train dataset contains triples
   that contain both the head entity h and the relation $r$, having the size of the number of non-zero elements
5. The index vector is now applied on the tail entity column of the train dataset, returning all tail entity IDs $t'$
   that combined with $h$ and $r$ lead to triples contained in the train dataset
6. Finally, the $t'$ tail entity ID index vector is applied on the initially mentioned vector returned by
   :func:`pykeen.models.base.Model.score_t` for all possible
   triples $(h, r, t')$ and all affected scores are set to ``float('nan')`` following the IEEE-754 specification, which
   makes these scores non-comparable, effectively leading to the score vector for all possible novel triples
   $(h, r, t') \in T^{\text{filtered}}_{h,r}$.

$H^{\text{filtered}}_{r,t}$ is obtained from $H_{r,t}$ in a similar fashion.

.. _sub_batching:

Sub-batching
------------
With growing model and dataset sizes the KGEM at hand is likely to exceed the memory provided by GPUs. Especially during
training it might be desired to train using a certain batch size. When this batch size is too big for the hardware at
hand, PyKEEN allows to set a sub-batch size in the range of :math:`[1, {batch size}]`. When the sub-batch size is set,
PyKEEN automatically accumulates the gradients after each sub-batch and clears the computational graph during training.
This allows to train KGEMs on GPU that otherwise would be too big for the hardware at hand, while the obtained results
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
above described process :ref:`sub_batching`. The batch sizes are determined using binary search taking into
consideration the `CUDA architecture <https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9926-tensor-core-performance-the-ultimate-guide.pdf>`_
which ensures that the chosen batch size is the most CUDA efficient one.
