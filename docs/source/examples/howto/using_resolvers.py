"""Configuring the loss via class-resolver."""

# %%
# A *subclass* of :class:`~pykeen.losses.Loss`, e.g., :class:`~pykeen.losses.MarginRankingLoss`, is instantiated
# with the given ``loss_kwargs`` as (keyword-based) parameters.
from pykeen.losses import MarginRankingLoss, loss_resolver

loss = MarginRankingLoss
loss_kwargs = None  # equivalent to {}
instance = loss_resolver.make(loss, pos_kwargs=loss_kwargs)
assert isinstance(instance, MarginRankingLoss)
assert instance.margin == MarginRankingLoss().margin

# %%
# We can also choose different instantiation parameters by:
loss_kwargs = {"margin": 2}
instance = loss_resolver.make(loss, pos_kwargs=loss_kwargs)
assert isinstance(instance, MarginRankingLoss)
assert instance.margin == 2
