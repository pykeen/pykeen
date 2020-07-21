Understanding the Evaluation
============================
This part of the tutorial is aimed to help you understand the evaluation of knowledge graph embeddings.
In particular it explains rank-based evaluation metrics reported in :class:`pykeen.evaluation.RankBasedMetricResults`.

Knowledge graph embedding are usually evaluated on the task of link prediction. To this end, an evaluation set of
triples :math:`\mathcal{T}_{eval} \subset \mathcal{E} \times \mathcal{R} \times \mathcal{E}` is provided, and for each
triple :math:`(h, r, t) \in \mathcal{T}_{eval}` in this set, we solve two tasks:

* In the *right-side* prediction task, we are given a par of head entity and relation and aim to predict the tail,
  i.e. :math:`(h, r, ?)`. To this end, we use the knowledge graph embedding model to *score* each of the possible
  choices :math:`(h, r, e)` for :math:`e \in \mathcal{E}`. Higher scores indicate higher plausibility.
* Analogously, in the *left-side* prediction task, we are provided a pair of relation and tail entity and aim to
  predict the head, i.e. :math:`(?, r, t)`. Again, each possible choice :math:`(e, r, t)` for
  :math:`e \in \mathcal{E}` is scored according to the knowledge graph embedding model.

.. note ::
    Practically, many embedding models allow fast computation of all scores :math:`(e, r, t)` for all
    :math:`e \in \mathcal{E}`, than just passing the triples through the model's score function. As an example,
    consider DistMult with the score function
    :math:`score(h,r,t)=\sum_{i=1}^d \mathbf{h}_i \cdot \mathbf{r}_i \cdot \mathbf{t}_i`. Here, we can score all
    entities as candidate heads for a given tail and relation by first computing the element-wise product of tail and
    relation, and then performing a matrix multiplication with the matrix of all entity embeddings.
    # TODO: Link to section explaining this concept.


Rank-Based Evaluation
---------------------
In the rank-based evaluation protocol, the scores are used to sort the list of possible choices by decreasing score,
and determine the *rank* of the true choice, i.e. the index in the sorted list. Smaller ranks indicate better
performance. Based on these individual ranks, which are obtained for each evaluation triple and each side of the
prediction (left/right), there exist several aggregation measures to quantify the performance of a model in a single
number.

While the aforementioned definition of the rank as "the index in the sorted list" is intuitive, it does not specify
what happens when there are multiple choices with exactly the same score. Therefore, in existing works, different
variants have been implemented, which yield different results in the presence of equal scores.

* The *optimistic* rank assumes that the true choice is on the first position of all those with equal score.
* The *pessimistic* rank assumes that the true choice is on the last position of all those with equal score.
* The *realistic* rank is the mean of the optimistic and the pessimistic rank, and moreover the expected value over
  all permutations respecting the sort order.
* The *non-deterministic* rank delegates the decision to the sort algorithm. Thus, the result depends on the internal
  tie breaking mechanism of the sort algorithm's implementation.

In pykeen, we support the first three: optimistic, pessimistic and realistic. When only using a single score, the
realistic score should be reported. The pessimistic and optimistic rank, or more specific the deviation between both,
can be used to detect whether a model predicts exactly equal scores for many choices. There are a few causes such as

* finite-precision arithmetic in conjunction with explicitly using sigmoid activation
* clamping of scores, e.g. by using a ReLU activation or similar.

Besides the different rank definitions, we also report scores for the individual side predictions, i.e.

* "head": The rank-based metric evaluated only for the head / left-side prediction.
* "tail": The rank-based metric evaluated only for the tail / right-side prediction.
* "both": The rank-based metric evaluated on both predictions.

By default, "both" is often used in publications. The side-specific scores can however often give access to
interesting insights, such as the difference in difficulty of predicting a head/tail given the rest, or the model's
incapability to solve of one the tasks.

Finally, the rank-based evaluation allows using the "filtered setting", which is enabled by default. When evaluating
the tail prediction for a triple :math:`(h, r, t)`, i.e. scoring all triples :math:`(h, r, e)`, there may be
additional known triples :math:`(h, r, t')` for :math:`t \neq t'`. If the model predicts a higher score for
:math:`(h, r, t')`, the rank will increase, and hence the measured model performance will decrease. However, giving
:math:`(h, r, t')` a high score (and thus a low rank) is desirable since it is a true triple as well. Thus, the
filtered evaluation setting ignores for a given triple :math:`(h, r, t)` the scores of all other *known* true triples
:math:`(h, r, t')`.