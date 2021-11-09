import numpy as np
import torch

from pykeen.losses import NSSALoss
from pykeen.datasets import WN18RR
from pykeen.models.unimodal.boxe_kg import BoxE
from pykeen.pipeline import pipeline

from torch.nn import functional


# TODO: Align optimizer settings: Constant LR

class NSSALossLogging(NSSALoss):
    def forward(
        self,
        pos_scores: torch.FloatTensor,
        neg_scores: torch.FloatTensor,
        neg_weights: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # copy of NSSALoss.forward
        neg_loss = functional.logsigmoid(-neg_scores - self.margin)
        neg_loss = neg_weights * neg_loss
        neg_loss = self._reduction_method(neg_loss)
        print("-", -neg_loss.item())

        pos_loss = functional.logsigmoid(self.margin + pos_scores)
        pos_loss = self._reduction_method(pos_loss)
        print("+", -pos_loss.item())

        loss = -pos_loss - neg_loss

        if self._reduction_method is torch.mean:
            loss = loss / 2.0

        return loss



def main():
    embedding_dim = 500
    unif_init_bound = 2 * np.sqrt(embedding_dim)
    init_kw = dict(a=-1 / unif_init_bound, b=1 / unif_init_bound)
    size_init_kw = dict(a=-1, b=1)
    dataset = WN18RR()
    triples_factory = dataset.training
    model = BoxE(
        triples_factory=triples_factory,
        embedding_dim=500,
        norm_order=2,
        tanh_map=True,
        entity_initializer=torch.nn.init.uniform_,
        entity_initializer_kwargs=init_kw,
        relation_initializer=torch.nn.init.uniform_,
        relation_initializer_kwargs=init_kw,
        relation_size_initializer=torch.nn.init.uniform_,
        relation_size_initializer_kwargs=size_init_kw,
    )

    results = pipeline(
        random_seed=1000000,
        dataset=dataset,
        model=model,
        training_kwargs=dict(num_epochs=300, batch_size=512, checkpoint_name="tria.pt", checkpoint_frequency=100),
        loss=NSSALossLogging(margin=5, adversarial_temperature=0.0, reduction="sum"),
        lr_scheduler=torch.optim.lr_scheduler.ConstantLR,
        lr_scheduler_kwargs=dict(total_iters=0),
        training_loop="sLCWA",
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=150),
        result_tracker="json",
        result_tracker_kwargs=dict(name="test.json"),
        evaluation_kwargs=dict(batch_size=16),
        optimizer=torch.optim.Adam,
        optimizer_kwargs=dict(lr=0.001)   # Cancel out the thing
    )

if __name__ == '__main__':
    main()
