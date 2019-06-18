import torch
from torch._jit_internal import weak_script_method
from torch.nn.modules.loss import _Loss


class SoftplusLoss(_Loss):

    def __init__(self, reduction='mean'):
        super(SoftplusLoss, self).__init__(reduction=reduction)
        self.softplus = torch.nn.Softplus(beta=1, threshold=20)

    @weak_script_method
    def forward(self, scores, labels):
        """."""
        labels = labels * (-1)
        scores = labels * scores

        loss = self.softplus(labels * scores)
        loss = torch.mean(loss) if self.reduction == 'mean' else torch.sum(scores)

        return loss
