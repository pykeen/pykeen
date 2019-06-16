
import torch
from torch._jit_internal import weak_script_method
from torch.nn.modules.loss import _Loss


class SoftplusLoss(_Loss):

    def __init__(self,reduction='mean'):
        super(SoftplusLoss, self).__init__(reduction=reduction)

    @weak_script_method
    def forward(self, scores, labels):
        torch.exponential_()

