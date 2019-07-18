from typing import Optional

import torch
from torch import nn
from torch.nn import functional

from poem.constants import GPU, SCORING_FUNCTION_NORM, TRANS_H_NAME, WEIGHT_SOFT_CONSTRAINT_TRANS_H
from poem.models.base import BaseModule
from poem.utils import slice_triples


class TransH(BaseModule):
    """An implementation of TransH [wang2014]_.

    This model extends TransE by applying the translation from head to tail entity in a relational-specific hyperplane.

    .. [wang2014] Wang, Z., *et al.* (2014). `Knowledge Graph Embedding by Translating on Hyperplanes
                  <https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546>`_. AAAI. Vol. 14.

    .. seealso::

       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/master/models/TransH.py
    """

    model_name = TRANS_H_NAME
    margin_ranking_loss_size_average: bool = False
    # FIXME why is this not summing the BaseModule.hyper_params?
    hyper_params = (SCORING_FUNCTION_NORM, WEIGHT_SOFT_CONSTRAINT_TRANS_H)

    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int = 50,
            scoring_fct_norm: int = 1,
            soft_weight_constraint: float = 0.05,
            criterion: nn.modules.loss = nn.MarginRankingLoss(margin=1., reduction='mean'),
            preferred_device: str = GPU,
            random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
        self.weighting_soft_constraint = soft_weight_constraint
        self.epsilon = nn.Parameter(torch.Tensor(0.005, requires_grad=True))
        self.scoring_fct_norm = scoring_fct_norm
        self.relation_embeddings = None
        self.normal_vector_embeddings = None

        self._init_embeddings()

    def _init_embeddings(self):
        super()._init_embeddings()
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.normal_vector_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        # TODO: Add initialization

    def project_to_hyperplane(self, entity_embs, normal_vec_embs):
        """

        :param entity_embs: Embeddings of entities with shape batch_size x 1 x embedding_dimension
        :param normal_vec_embs: Normal vectors with shape batch_size x 1 x embedding_dimension
        :return: Projected entities of shape batch_size x embedding_dim
        """
        scaling_factors = torch.sum(normal_vec_embs * entity_embs, dim=-1).unsqueeze(1)
        heads_projected_on_normal_vecs = scaling_factors * normal_vec_embs
        projections = (entity_embs - heads_projected_on_normal_vecs).view(-1, self.embedding_dim)
        return projections

    def _compute_mr_loss(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        mrl_loss = super()._compute_mr_loss(positive_scores, negative_scores)
        soft_constraint_loss = self.compute_soft_constraint_loss()
        loss = mrl_loss + soft_constraint_loss
        return loss

    def compute_soft_constraint_loss(self):
        """Compute the soft constraints."""
        norm_of_entities = torch.norm(self.entity_embeddings.weight, p=2, dim=1)
        square_norms_entities = torch.mul(norm_of_entities, norm_of_entities)
        entity_constraint = square_norms_entities - self.num_entities * 1.
        entity_constraint = torch.abs(entity_constraint)
        entity_constraint = torch.sum(entity_constraint)

        orthogonalty_constraint_numerator = torch.mul(
            self.normal_vector_embeddings.weight,
            self.relation_embeddings.weight,
        )
        orthogonalty_constraint_numerator = torch.sum(orthogonalty_constraint_numerator, dim=1)
        orthogonalty_constraint_numerator = torch.mul(
            orthogonalty_constraint_numerator,
            orthogonalty_constraint_numerator,
        )

        orthogonalty_constraint_denominator = torch.norm(self.relation_embeddings.weight, p=2, dim=1)
        orthogonalty_constraint_denominator = torch.mul(
            orthogonalty_constraint_denominator,
            orthogonalty_constraint_denominator,
        )

        orthogonalty_constraint = (orthogonalty_constraint_numerator / orthogonalty_constraint_denominator) - \
                                  (self.num_relations * self.epsilon)
        orthogonalty_constraint = torch.abs(orthogonalty_constraint)
        orthogonalty_constraint = torch.sum(orthogonalty_constraint)

        soft_constraints_loss = self.weighting_soft_constraint * (entity_constraint + orthogonalty_constraint)

        return soft_constraints_loss

    def forward_owa(self, triples):
        if not self.forward_constraint_applied:
            self.apply_forward_constraints()
        heads, relations, tails = slice_triples(triples)
        head_embeddings = self._get_embeddings(
            elements=heads, embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim,
        )

        relation_embeddings = self._get_embeddings(
            elements=relations,
            embedding_module=self.relation_embeddings,
            embedding_dim=self.embedding_dim,
        )
        tail_embeddings = self._get_embeddings(
            elements=tails,
            embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim,
        )
        normal_vec_embs = self._get_embeddings(
            elements=relations,
            embedding_module=self.normal_vector_embeddings,
            embedding_dim=self.embedding_dim,
        )
        head_embeddings = head_embeddings.view(-1, 1, self.embedding_dim)
        tail_embeddings = tail_embeddings.view(-1, 1, self.embedding_dim)
        normal_vec_embs = normal_vec_embs.view(-1, 1, self.embedding_dim)

        projected_heads = self.project_to_hyperplane(entity_embs=head_embeddings, normal_vec_embs=normal_vec_embs)
        projected_tails = self.project_to_hyperplane(entity_embs=tail_embeddings, normal_vec_embs=normal_vec_embs)

        sum_res = projected_heads + relation_embeddings - projected_tails
        norms = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        scores = - torch.mul(norms, norms)
        return scores

    # TODO: Implement forward_cwa

    def apply_forward_constraints(self):
        # Normalise the normal vectors by their l2 norms
        functional.normalize(self.normal_vector_embeddings.weight.data, out=self.normal_vector_embeddings.weight.data)
        self.forward_constraint_applied = True
