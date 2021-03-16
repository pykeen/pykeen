# -*- coding: utf-8 -*-

"""Riemannian SGD optimizer

Code originally written by Ivana Balažević
https://github.com/ibalazevic/multirelational-poincare/blob/master/rsgd.py
and used under the MIT License.
"""

import torch
from torch.optim.optimizer import Optimizer, required

from ..utils import full_p_exp_map

__all__ = [
    'RiemannianSGD',
]


class RiemannianSGD(Optimizer):
    def __init__(self, params, lr=required, param_names=None):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.param_names = [] if param_names is None else param_names

    def step(self, lr=None):
        loss = None
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if lr is None:
                    lr = group["lr"]
                if self.param_names[i] in ["Eh.weight", "rvh.weight"]:
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
