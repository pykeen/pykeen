# -*- coding: utf-8 -*-

"""Riemannian SGD optimizer.

Code originally written by Ivana Balažević
https://github.com/ibalazevic/multirelational-poincare/blob/master/rsgd.py
and used under the MIT License.
"""
from typing import Sequence, Set

import torch
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer, required

from ..utils import full_p_exp_map

__all__ = [
    'RiemannianSGD',
]


class RiemannianSGD(Optimizer):
    """A variant of :class:`torch.optim.SGD` generalized for riemannian manifolds."""

    def __init__(self, params: Sequence[Parameter], param_names: Sequence[str], poincare: Set[str], lr=required):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self._pykeen_extras = {
            'param_names': param_names,
            'poincare': poincare,
        }

    def is_poincare(self, param) -> bool:
        """Check if the param is a poincare param."""
        return self._pykeen_extras['param_names'][id(param)] in self._pykeen_extras['poincare']

    def step(self, lr=None):  # noqa:D102
        loss = None
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if lr is None:
                    lr = group["lr"]

                if self.is_poincare(p):
                    d_p = poincare_grad(p, d_p)
                    p.data = poincare_update(p, d_p, lr)
                else:
                    p.data = euclidean_update(p, d_p, lr)
        return loss


def euclidean_update(p, d_p, lr):
    p.data = p.data - lr * d_p
    return p.data


def poincare_grad(p, d_p):
    p_sqnorm = torch.clamp(torch.sum(p.data ** 2, dim=-1, keepdim=True), 0, 1 - 1e-5)
    d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
    return d_p


def poincare_update(p, d_p, lr):
    v = -lr * d_p
    p.data = full_p_exp_map(p.data, v)
    return p.data
