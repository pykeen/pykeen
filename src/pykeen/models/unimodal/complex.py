# -*- coding: utf-8 -*-

"""Implementation of the ComplEx model."""

from typing import Optional

import torch
import torch.nn as nn

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss, SoftplusLoss
from ...regularizers import LpRegularizer, Regularizer
from ...triples import TriplesFactory
from ...utils import get_embedding_in_canonical_shape, split_complex

__all__ = [
    'ComplEx',
]


class ComplEx(EntityRelationEmbeddingModel):
    """An implementation of ComplEx [trouillon2016]_.

    .. seealso ::
        Official implementation: https://github.com/ttrouill/complex/

    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=300, q=50),
    )
    #: The default loss function class
    loss_default = SoftplusLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs = dict(reduction='mean')
    #: The regularizer used by [trouillon2016]_ for ComplEx.
    regularizer_default = LpRegularizer
    #: The LP settings used by [trouillon2016]_ for ComplEx.
    regularizer_default_kwargs = dict(
        weight=0.01,
        p=2.0,
        normalize=True,
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 200,
        automatic_memory_optimization: Optional[bool] = None,
        loss: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        """Initialize the module.

        :param triples_factory: TriplesFactory
            The triple factory connected to the model.
        :param embedding_dim: int
            The embedding dimensionality of the entity embeddings.
        :param automatic_memory_optimization: bool
            Whether to automatically optimize the sub-batch size during training and batch size during evaluation with
            regards to the hardware at hand.
        :param loss: OptionalLoss (optional)
            The loss to use. Defaults to SoftplusLoss.
        :param preferred_device: str (optional)
            The default device where to model is located.
        :param random_seed: int (optional)
            An optional random seed to set before the initialization of weights.
        :param regularizer: BaseRegularizer
            The regularizer to use.
        """
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=2 * embedding_dim,  # complex embeddings
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        # Finalize initialization
        self.reset_parameters_()

    def _reset_parameters_(self):  # noqa: D102
        # initialize with entity and relation embeddings with standard normal distribution, cf.
        # https://github.com/ttrouill/complex/blob/dc4eb93408d9a5288c986695b58488ac80b1cc17/efe/models.py#L481-L487
        nn.init.normal_(tensor=self.entity_embeddings.weight, mean=0., std=1.)
        nn.init.normal_(tensor=self.relation_embeddings.weight, mean=0., std=1.)

    @staticmethod
    def interaction_function(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the interaction function of ComplEx for given embeddings.

        The embeddings have to be in a broadcastable shape.

        :param h:
            Head embeddings.
        :param r:
            Relation embeddings.
        :param t:
            Tail embeddings.

        :return: shape: (...)
            The scores.
        """
        # split into real and imaginary part
        (h_re, h_im), (r_re, r_im), (t_re, t_im) = [split_complex(x=x) for x in (h, r, t)]

        # ComplEx space bilinear product
        # *: Elementwise multiplication
        return sum(
            (hh * rr * tt).sum(dim=-1)
            for hh, rr, tt in [
                (h_re, r_re, t_re),
                (h_re, r_im, t_im),
                (h_im, r_re, t_im),
                (h_im, r_im, t_re),
            ]
        )

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # get embeddings
        h, r, t = [
            get_embedding_in_canonical_shape(embedding=e, ind=ind)
            for e, ind in [
                (self.entity_embeddings, hrt_batch[:, 0]),
                (self.relation_embeddings, hrt_batch[:, 1]),
                (self.entity_embeddings, hrt_batch[:, 2]),
            ]
        ]

        # Compute scores
        scores = self.interaction_function(h=h, r=r, t=t)

        # Regularization
        self.regularize_if_necessary(h, r, t)

        return scores
