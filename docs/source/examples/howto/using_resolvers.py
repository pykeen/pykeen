"""Configuring the loss via class-resolver."""

# %%
from pykeen.losses import MarginRankingLoss, loss_resolver

loss = MarginRankingLoss
loss_kwargs = None  # equivalent to {}
instance = loss_resolver.make(loss, pos_kwargs=loss_kwargs)
assert isinstance(instance, MarginRankingLoss)
assert instance.margin == MarginRankingLoss().margin

# %%
loss_kwargs = {"margin": 2}
instance = loss_resolver.make(loss, pos_kwargs=loss_kwargs)
assert isinstance(instance, MarginRankingLoss)
assert instance.margin == 2
