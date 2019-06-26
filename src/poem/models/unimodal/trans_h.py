import torch
from torch import nn

from poem.constants import TRANS_H_NAME, SCORING_FUNCTION_NORM, WEIGHT_SOFT_CONSTRAINT_TRANS_H, GPU
from poem.models.base_owa import BaseOWAModule, slice_triples


class TransH(BaseOWAModule):
    """An implementation of TransH [wang2014]_.

    This model extends TransE by applying the translation from head to tail entity in a relational-specific hyperplane.

    .. [wang2014] Wang, Z., *et al.* (2014). `Knowledge Graph Embedding by Translating on Hyperplanes
                  <https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546>`_. AAAI. Vol. 14.

    .. seealso::

       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/master/models/TransH.py
    """

    model_name = TRANS_H_NAME
    margin_ranking_loss_size_average: bool = False
    hyper_params = [SCORING_FUNCTION_NORM, WEIGHT_SOFT_CONSTRAINT_TRANS_H]

    def __init__(self, num_entities, num_relations, embedding_dim=50, scoring_fct_norm=1, soft_weight_constraint=0.05,
                 criterion=nn.MarginRankingLoss(margin=1., reduction='mean'), preferred_device=GPU) -> None:
        super(TransH, self).__init__(num_entities, num_relations, criterion, embedding_dim, preferred_device)

        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.normal_vector_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.weighting_soft_constraint = soft_weight_constraint

        self.epsilon = nn.Parameter(torch.tensor(0.005, requires_grad=True))
        self.scoring_fct_norm = scoring_fct_norm
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
        """."""
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

        orthogonalty_constraint_numerator = torch.mul(self.normal_vector_embeddings.weight,
                                                      self.relation_embeddings.weight)
        orthogonalty_constraint_numerator = torch.sum(orthogonalty_constraint_numerator, dim=1)
        orthogonalty_constraint_numerator = torch.mul(orthogonalty_constraint_numerator,
                                                      orthogonalty_constraint_numerator)

        orthogonalty_constraint_denominator = torch.norm(self.relation_embeddings.weight, p=2, dim=1)
        orthogonalty_constraint_denominator = torch.mul(orthogonalty_constraint_denominator,
                                                        orthogonalty_constraint_denominator)

        orthogonalty_constraint = (orthogonalty_constraint_numerator / orthogonalty_constraint_denominator) - \
                                  (self.num_relations * self.epsilon)
        orthogonalty_constraint = torch.abs(orthogonalty_constraint)
        orthogonalty_constraint = torch.sum(orthogonalty_constraint)

        soft_constraints_loss = self.weighting_soft_constraint * (entity_constraint + orthogonalty_constraint)

        return soft_constraints_loss

    def _score_triples(self, triples):
        """"""
        heads, relations, tails = slice_triples(triples)
        head_embeddings = self._get_embeddings(
            elements=heads, embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim
        )

        relation_embeddings = self._get_embeddings(
            elements=relations,
            embedding_module=self.relation_embeddings,
            embedding_dim=self.embedding_dim
        )
        tail_embeddings = self._get_embeddings(
            elements=tails,
            embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim
        )
        normal_vec_embs = self._get_embeddings(
            elements=relations,
            embedding_module=self.normal_vector_embeddings,
            embedding_dim=self.embedding_dim
        )
        scores = self._compute_scores(head_embeddings, relation_embeddings, tail_embeddings, normal_vec_embs)

        return scores

    def _compute_scores(self, head_embeddings, relation_embeddings, tail_embeddings, normal_vec_embs):
        """"""
        head_embeddings = head_embeddings.view(-1, 1, self.embedding_dim)
        tail_embeddings = tail_embeddings.view(-1, 1, self.embedding_dim)
        normal_vec_embs = normal_vec_embs.view(-1, 1, self.embedding_dim)

        projected_heads = self.project_to_hyperplane(entity_embs=head_embeddings, normal_vec_embs=normal_vec_embs)
        projected_tails = self.project_to_hyperplane(entity_embs=tail_embeddings, normal_vec_embs=normal_vec_embs)

        sum_res = projected_heads + relation_embeddings - projected_tails
        norms = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        scores = - torch.mul(norms, norms)

        return scores

    def apply_forward_constraints(self):
        """."""
        # Normalise the normal vectors by their l2 norms
        norms = torch.norm(self.normal_vector_embeddings.weight, p=2, dim=1).data
        self.normal_vector_embeddings.weight.data = self.normal_vector_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.normal_vector_embeddings.weight))
