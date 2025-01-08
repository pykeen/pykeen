.. _interactions:

Interaction Functions
=====================

In PyKEEN, an *interaction function* refers to a function that maps *representations* for head entities, relations, and tail entities to a scalar plausibility score. In the simplest case, head entities, relations, and tail entities are each represented by a single tensor. However, there are also interaction functions that use multiple tensors, e.g. :class:`~pykeen.nn.modules.NTNInteraction`.

Interaction functions can also have trainable parameters that are global and not related to a single entity or relation. An example is :class:`~pykeen.nn.modules.TuckERInteraction` with its core tensor. We call such functions stateful and all others stateless.

.. todo::
    - general description, larger is better
    - stateful vs. state-less, extra parameters
    - norm-based / semantic matching & factorization / neural
    - value ranges?
    - properties? (symmetric, etc.)
    - computational complexity?

Base
----
- :class:`~pykeen.nn.modules.Interaction`
- :class:`~pykeen.nn.modules.FunctionalInteraction`
- :class:`~pykeen.nn.modules.NormBasedInteraction`

Combinations & Adapters
-----------------------
The :class:`~pykeen.nn.modules.DirectionAverageInteraction` calculates a plausibility by averaging the plausibility scores of a base function over the forward and backward representations.
It can be seen as a generalization of :class:`~pykeen.nn.modules.SimplEInteraction`.

The :class:`~pykeen.nn.modules.MonotonicAffineTransformationInteraction` adds trainable scalar scale and bias terms to an existing interaction function. The scale parameter is parametrized to take only positive values, preserving the interpretation of larger values corresponding to more plausible triples.
This adapter is particularly useful for base interactions with a restricted range of values, such as norm-based interactions, and loss functions with absolute decision thresholds, such as point-wise losses, e.g., :class:`~pykeen.losses.BCEWithLogitsLoss`.

The :class:`~pykeen.nn.modules.ClampedInteraction` constrains the scores to a given range of values. While this ensures that scores cannot exceed the bounds, using :func:`torch.clamp()` also means that no gradients are propagated for inputs with out-of-bounds scores. It can also lead to tied scores during evaluation, which can cause problems with some variants of the score functions, see :ref:`understanding-evaluation`.


Concrete Interactions
---------------------

- :class:`~pykeen.nn.modules.AutoSFInteraction`
- :class:`~pykeen.nn.modules.BoxEInteraction`
- :class:`~pykeen.nn.modules.ComplExInteraction`
- :class:`~pykeen.nn.modules.ConvEInteraction`
- :class:`~pykeen.nn.modules.ConvKBInteraction`
- :class:`~pykeen.nn.modules.CPInteraction`
- :class:`~pykeen.nn.modules.CrossEInteraction`
- :class:`~pykeen.nn.modules.DistMAInteraction`
- :class:`~pykeen.nn.modules.DistMultInteraction`
- :class:`~pykeen.nn.modules.ERMLPInteraction`
- :class:`~pykeen.nn.modules.ERMLPEInteraction`
- :class:`~pykeen.nn.modules.HolEInteraction`
- :class:`~pykeen.nn.modules.KG2EInteraction`
- :class:`~pykeen.nn.modules.LineaREInteraction`
- :class:`~pykeen.nn.modules.MultiLinearTuckerInteraction`
- :class:`~pykeen.nn.modules.MuREInteraction`
- :class:`~pykeen.nn.modules.NTNInteraction`
- :class:`~pykeen.nn.modules.PairREInteraction`
- :class:`~pykeen.nn.modules.ProjEInteraction`
- :class:`~pykeen.nn.modules.QuatEInteraction`
- :class:`~pykeen.nn.modules.RESCALInteraction`
- :class:`~pykeen.nn.modules.RotatEInteraction`
- :class:`~pykeen.nn.modules.SEInteraction`
- :class:`~pykeen.nn.modules.SimplEInteraction`
- :class:`~pykeen.nn.modules.TorusEInteraction`
- :class:`~pykeen.nn.modules.TransDInteraction`
- :class:`~pykeen.nn.modules.TransEInteraction`
- :class:`~pykeen.nn.modules.TransFInteraction`
- :class:`~pykeen.nn.modules.TransformerInteraction`
- :class:`~pykeen.nn.modules.TransHInteraction`
- :class:`~pykeen.nn.modules.TransRInteraction`
- :class:`~pykeen.nn.modules.TripleREInteraction`
- :class:`~pykeen.nn.modules.TuckERInteraction`
- :class:`~pykeen.nn.modules.UMInteraction`
