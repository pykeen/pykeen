.. _normalizer_constrainer_regularizer:

Normalizer, Constrainer & Regularizer
=====================================

Sometimes we want to impose constraints on parameters such as the embedding vector. This can be motivated by geometric
importance of individual parameters (e.g., acting as a normal vector), indicating a preference for one solution over
many otherwise equivalent solutions, or acting as a general regularization term to reduce the risk of overfitting.

There are several ways to impose these constraints. To better differentiate between them, PyKEEN distinguishes between
*normalizers*, *constrainers*, and *regularizers*.

Normalizer
----------

A *normalizer* is essentially a re-parameterization: For instance, to normalize vector $\mathbf{x} \in \mathbb{R}^d$ to
unit norm, we introduce a new variable $\overline{\mathbf{x}} = \frac{1}{\|\mathbf{x}\|}\mathbf{x}$, and use this
variable instead of $\mathbf{x}$.

This normalization is applied within the computational graph, so gradients for $\mathbf{x}$ will include the effect of
the normalization constant, effectively canceling out any contribution that would not change the direction of the
gradient, but rather re-scale its length.

Normalizers are strict, i.e. the normalized variable is guaranteed to satisfy the constraint.

Since normalizers are part of the computational graph, they are usually implemented directly as part of a module, e.g.,
in :class:`pykeen.nn.representation.Embedding`.

Constrainer
-----------

A *constrainer* applies the same strict normalization, *but outside of the computational graph*. So we do the gradient
computation without respecting the constraint, and only after applying the optimization update, e.g., we do a
renormalization.

This method can be seen as a `relaxation <https://en.wikipedia.org/wiki/Relaxation_(approximation)>`_ of the constrained
problem that ignores the constraints and projects the (approximate) solution of the unconstrained problem back into the
constraint set.

Constrainers guarantee the constraint *after each gradient step*, but may violate the constaints in between.

Constrainers are applied in :meth:`pykeen.nn.representation.Representation.post_parameter_update`.

.. warning::

    If you use PyKEEN modules outside of PyKEEN's training methods, you need to make sure to call
    :meth:`pykeen.nn.representation.Representation.post_parameter_update` yourself after each parameter update.

.. seealso::

    - `Proximal gradient methods <https://en.wikipedia.org/wiki/Proximal_gradient_method>`_

Regularizer
-----------

A *regularizer* does not impose strict normalization, but rather adds a term to the loss function that penalizes
violation of the constraint. This allows constraints to be (severely) violated in the early stages of the optimization,
when the original loss term is still high and thus the relative contribution of the regularization term is small.
However, once the optimization has found a sufficiently good solution for the original loss term, it can only further
improve the total loss by also respecting the regularization term.

You can find implementations of regularizers by looking at subclasses of :class:`pykeen.regularizers.Regularizer`.

.. seealso::

    - A related concept in linear optimization are `slack variables <https://en.wikipedia.org/wiki/Slack_variable>`_.
