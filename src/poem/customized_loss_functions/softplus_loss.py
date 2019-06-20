import torch
from torch._jit_internal import weak_script_method
from torch.nn.modules.loss import _Loss


class SoftplusLoss(_Loss):

    def __init__(self, reduction='mean'):
        super(SoftplusLoss, self).__init__(reduction=reduction)
        self.softplus = torch.nn.Softplus(beta=1, threshold=20)
        if self.reduction == 'mean':
            self._reduction_method = torch.mean
        else:
            self._reduction_method = torch.sum

    @weak_script_method
    def forward(self, scores, labels):
        """."""
        labels = labels * (-1)
        scores = labels * scores
        # FIXME Why not just do loss = self.softplus(labels * labels * scores * -1)?
        loss = self.softplus(labels * scores)
        loss = self._reduction_method(loss)
        return loss
