Loss Functions
==============
.. automodapi:: pykeen.losses
    :no-heading:
    :headings: --

PyTorch Losses
--------------
There are several loss functions from PyTorch that can be used
as well:

===============  ==========================================
Name             Reference
===============  ==========================================
bce              :class:`torch.nn.BCELoss`
marginranking    :class:`torch.nn.MarginRankingLoss`
mse              :class:`torch.nn.MSELoss`
===============  ==========================================

HPO Defaults
------------
.. autodata:: losses_hpo_defaults

Lookup
------
.. autofunction:: get_loss_cls
