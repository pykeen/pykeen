"""Implementing your own interaction modules."""

# %%
# To implement TransE in PyKEEN, you need to subclass the :class:`~pykeen.nn.modules.Interaction`. This class is
# itself a subclass of :class:`torch.nn.Module`, which means that you need to provide an implementation of
# :meth:`torch.nn.Module.forward`. However, the arguments are predefined as ``h``, ``r``, and ``t``, which
# correspond to the representations of the head, relation, and tail, respectively.
from pykeen.nn.modules import Interaction


class TransEInteraction(Interaction):
    """A simplified re-implementation of TransE."""

    def forward(self, h, r, t):
        """Score a batch of head, relation, and tail representations."""
        return -(h + r - t).norm(p=2, dim=-1)


# %%
# As a researcher who just invented TransE, you might wonder what would happen if you replaced the addition ``+``
# with multiplication ``*``. You might then end up with a new interaction like this (which just happens to be
# DistMult, which was published just a year after TransE):
class DistMultInteraction(Interaction):
    """A simplified re-implementation of DistMult."""

    def forward(self, h, r, t):
        """Score a batch of head, relation, and tail representations."""
        return (h * r * t).sum(dim=-1)


# %%
# While we previously defined TransE with the L2 norm, it could be calculated with a different value for p. This
# could be incorporated into the interaction definition by using ``__init__()``, storing the value for p in the
# instance, then accessing it in ``forward()``.
class TransEInteractionWithNorm(Interaction):
    """A simplified re-implementation of TransE with a configurable norm."""

    def __init__(self, p: int):
        """Initialize the interaction with the given norm."""
        super().__init__()
        self.p = p

    def forward(self, h, r, t):
        """Score a batch of head, relation, and tail representations."""
        return -(h + r - t).norm(p=self.p, dim=-1)


# %%
# In ER-MLP, the multi-layer perceptron consists of an input layer, a hidden layer, and an output layer. The input
# is represented by the concatenation embeddings of the heads, relations and tail embeddings. Global trainable
# parameters, unlike hyper-parameters, can also be defined in the ``__init__()`` function of your
# :class:`~pykeen.nn.modules.Interaction` class. They are trained jointly with the entity and relation embeddings
# during training.
import torch.nn


class ERMLPInteraction(Interaction):
    """A simplified re-implementation of ER-MLP."""

    def __init__(self, embedding_dim: int, hidden_dim: int):
        """Initialize the interaction's multi-layer perceptron."""
        super().__init__()
        # The weights of this MLP will be learned.
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=3 * embedding_dim, out_features=hidden_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=1, bias=True),
        )

    def forward(self, h, r, t):
        """Score a batch of head, relation, and tail representations."""
        x = torch.cat([h, r, t], dim=-1)
        return self.mlp(x)


# %%
# The Structured Embedding uses a 2-tensor for representing each relation. For the purposes of this tutorial, we
# will propose a simplification to Structured Embedding (also similar to TransR) where the same relation 2-tensor
# is used to project both the head and tail entities.
class SimplifiedStructuredEmbeddingInteraction(Interaction):
    """A simplified re-implementation of Structured Embedding."""

    relation_shape = ("dd",)

    def forward(self, h, r, t):
        """Score a batch of head, relation, and tail representations."""
        h_proj = r @ h.unsqueeze(dim=-1)
        t_proj = r @ t.unsqueeze(dim=-1)
        return -(h_proj - t_proj).squeeze(dim=-1).norm(p=2, dim=-1)


# %%
# Sometimes, like in the canonical version of Structured Embedding, you need more than one representation for
# entities and/or relations. To specify this, you just need to extend the tuple for ``relation_shape`` with more
# entries, each corresponding to the sequence of representations.
class StructuredEmbeddingInteraction(Interaction):
    """A re-implementation of Structured Embedding."""

    relation_shape = (
        "dd",  # Corresponds to the head projection matrix
        "dd",  # Corresponds to the tail projection matrix
    )

    def forward(self, h, r, t):
        """Score a batch of head, relation, and tail representations."""
        # Since the relation_shape is more than length 1, the r value is given as a sequence
        # of the representations defined there. You can use tuple unpacking to get them out
        r_h, r_t = r
        h_proj = r_h @ h.unsqueeze(dim=-1)
        t_proj = r_t @ t.unsqueeze(dim=-1)
        return -(h_proj - t_proj).squeeze(dim=-1).norm(p=2, dim=-1)


# %%
# TransD is an example of an interaction module that not only uses two different representations for each entity
# and two representations for each relation, but they are of different dimensions. It can be implemented by
# choosing a different letter for use in the ``entity_shape`` and/or ``relation_shape``.
from pykeen.utils import project_entity


class TransDInteraction(Interaction):
    """A re-implementation of TransD."""

    entity_shape = ("d", "d")
    relation_shape = ("e", "e")

    def forward(self, h, r, t):
        """Score a batch of head, relation, and tail representations."""
        h, h_proj = h
        r, r_proj = r
        t, t_proj = t
        h_bot = project_entity(
            e=h,
            e_p=h_proj,
            r_p=r_proj,
        )
        t_bot = project_entity(
            e=t,
            e_p=t_proj,
            r_p=r_proj,
        )
        return -(h_bot + r - t_bot).norm(p=2, dim=-1)
