"""Implementing your own interaction modules."""

# %%
from pykeen.nn.modules import Interaction


class TransEInteraction(Interaction):
    """A simplified re-implementation of TransE."""

    def forward(self, h, r, t):
        """Score a batch of head, relation, and tail representations."""
        return -(h + r - t).norm(p=2, dim=-1)


# %%
class DistMultInteraction(Interaction):
    """A simplified re-implementation of DistMult."""

    def forward(self, h, r, t):
        """Score a batch of head, relation, and tail representations."""
        return (h * r * t).sum(dim=-1)


# %%
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
class SimplifiedStructuredEmbeddingInteraction(Interaction):
    """A simplified re-implementation of Structured Embedding."""

    relation_shape = ("dd",)

    def forward(self, h, r, t):
        """Score a batch of head, relation, and tail representations."""
        h_proj = r @ h.unsqueeze(dim=-1)
        t_proj = r @ t.unsqueeze(dim=-1)
        return -(h_proj - t_proj).squeeze(dim=-1).norm(p=2, dim=-1)


# %%
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
