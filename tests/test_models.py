# -*- coding: utf-8 -*-

"""Test KGE models"""

import unittest

from pykeen.constants import *
from pykeen.kge_models import TransE, TransH, TransR, TransD, DistMult, ERMLP, StructuredEmbedding, UnstructuredModel, \
    RESCAL, ConvE

import torch

TRANS_E_CONFIG = {
    NUM_ENTITIES: 5,
    NUM_RELATIONS: 5,
    EMBEDDING_DIM: 5,
    NORM_FOR_NORMALIZATION_OF_ENTITIES: 2,
    SCORING_FUNCTION_NORM: 1,
    MARGIN_LOSS: 4,
}

TRANS_H_CONFIG = {
    NUM_ENTITIES: 5,
    NUM_RELATIONS: 5,
    EMBEDDING_DIM: 5,
    NORM_FOR_NORMALIZATION_OF_ENTITIES: 2,
    SCORING_FUNCTION_NORM: 1,
    MARGIN_LOSS: 4,
    WEIGHT_SOFT_CONSTRAINT_TRANS_H: 0.05,
}

TRANS_R_CONFIG = {
    NUM_ENTITIES: 5,
    NUM_RELATIONS: 5,
    EMBEDDING_DIM: 5,
    RELATION_EMBEDDING_DIM: 3,
    SCORING_FUNCTION_NORM: 1,
    MARGIN_LOSS: 4,
}

TRANS_D_CONFIG = {
    NUM_ENTITIES: 5,
    NUM_RELATIONS: 5,
    EMBEDDING_DIM: 5,
    RELATION_EMBEDDING_DIM: 3,
    SCORING_FUNCTION_NORM: 1,
    MARGIN_LOSS: 4,
}

DISTMULT_CONFIG = {
    NUM_ENTITIES: 5,
    NUM_RELATIONS: 5,
    EMBEDDING_DIM: 5,
    SCORING_FUNCTION_NORM: 1,
    MARGIN_LOSS: 4,
}

ERMLP_CONFIG = {
    NUM_ENTITIES: 5,
    NUM_RELATIONS: 5,
    EMBEDDING_DIM: 2,
    SCORING_FUNCTION_NORM: 1,
    MARGIN_LOSS: 4,
}

SE_CONFIG = {
    NUM_ENTITIES: 5,
    NUM_RELATIONS: 5,
    EMBEDDING_DIM: 5,
    NORM_FOR_NORMALIZATION_OF_ENTITIES: 2,
    SCORING_FUNCTION_NORM: 1,
    MARGIN_LOSS: 4,
}

UM_CONFIG = {
    NUM_ENTITIES: 5,
    NUM_RELATIONS: 5,
    EMBEDDING_DIM: 5,
    NORM_FOR_NORMALIZATION_OF_ENTITIES: 2,
    SCORING_FUNCTION_NORM: 1,
    MARGIN_LOSS: 4,
}

RESCAL_CONFIG = {
    NUM_ENTITIES: 5,
    NUM_RELATIONS: 5,
    EMBEDDING_DIM: 5,
    NORM_FOR_NORMALIZATION_OF_ENTITIES: 2,
    SCORING_FUNCTION_NORM: 1,
    MARGIN_LOSS: 4,
}

CONV_E_CONFIG = {
    NUM_ENTITIES: 5,
    NUM_RELATIONS: 5,
    EMBEDDING_DIM: 5,
    CONV_E_INPUT_CHANNELS: 1,
    CONV_E_OUTPUT_CHANNELS: 2,
    CONV_E_KERNEL_HEIGHT: 5,
    CONV_E_KERNEL_WIDTH: 1,
    CONV_E_INPUT_DROPOUT: 0.3,
    CONV_E_OUTPUT_DROPOUT: 0.5,
    CONV_E_FEATURE_MAP_DROPOUT: 0.2,
    CONV_E_HEIGHT: 5,
    CONV_E_WIDTH: 1,
}

TEST_TRIPLES = torch.tensor([[0, 1, 1], [0, 1, 2]], dtype=torch.long)


class TestModelInstantiation(unittest.TestCase):
    """Test that all models can be instantiated."""

    def test_instantiate_trans_e(self):
        """Test that TransE can be instantiated."""
        trans_e = TransE(config=TRANS_E_CONFIG)
        self.assertIsNotNone(trans_e)
        self.assertEqual(trans_e.num_entities, 5)
        self.assertEqual(trans_e.num_relations, 5)
        self.assertEqual(trans_e.embedding_dim, 5)
        self.assertEqual(trans_e.l_p_norm_entities, 2)
        self.assertEqual(trans_e.scoring_fct_norm, 1)
        self.assertEqual(trans_e.margin_loss, 4)

    def test_instantiate_trans_h(self):
        """Test that TransH can be instantiated."""
        trans_h = TransH(config=TRANS_H_CONFIG)
        self.assertIsNotNone(trans_h)
        self.assertEqual(trans_h.num_entities, 5)
        self.assertEqual(trans_h.num_relations, 5)
        self.assertEqual(trans_h.embedding_dim, 5)
        self.assertEqual(trans_h.weighting_soft_constraint, 0.05)
        self.assertEqual(trans_h.scoring_fct_norm, 1)
        self.assertEqual(trans_h.margin_loss, 4)

    def test_instantiate_trans_r(self):
        """Test that TransR can be instantiated."""
        trans_r = TransR(config=TRANS_R_CONFIG)
        self.assertIsNotNone(trans_r)
        self.assertEqual(trans_r.num_entities, 5)
        self.assertEqual(trans_r.num_relations, 5)
        self.assertEqual(trans_r.embedding_dim, 5)
        self.assertEqual(trans_r.relation_embedding_dim, 3)
        self.assertEqual(trans_r.scoring_fct_norm, 1)
        self.assertEqual(trans_r.margin_loss, 4)

    def test_instantiate_trans_d(self):
        """Test that TransD can be instantiated."""
        trans_d = TransD(config=TRANS_D_CONFIG)
        self.assertIsNotNone(trans_d)
        self.assertEqual(trans_d.num_entities, 5)
        self.assertEqual(trans_d.num_relations, 5)
        self.assertEqual(trans_d.embedding_dim, 5)
        self.assertEqual(trans_d.relation_embedding_dim, 3)
        self.assertEqual(trans_d.scoring_fct_norm, 1)
        self.assertEqual(trans_d.margin_loss, 4)

    def test_instantiate_distmult(self):
        """Test that DistMult can be instantiated."""
        distmult = DistMult(config=DISTMULT_CONFIG)
        self.assertIsNotNone(distmult)
        self.assertEqual(distmult.num_entities, 5)
        self.assertEqual(distmult.num_relations, 5)
        self.assertEqual(distmult.embedding_dim, 5)
        self.assertEqual(distmult.margin_loss, 4)

    def test_instantiate_ermlp(self):
        """Test that ERMLP can be instantiated."""
        ermlp = ERMLP(config=ERMLP_CONFIG)
        self.assertIsNotNone(ermlp)
        self.assertEqual(ermlp.num_entities, 5)
        self.assertEqual(ermlp.num_relations, 5)
        self.assertEqual(ermlp.embedding_dim, 2)
        self.assertEqual(ermlp.margin_loss, 4)

    def test_instantiate_strcutured_embedding(self):
        """Test that StructuredEmbedding can be instantiated."""
        se = StructuredEmbedding(config=SE_CONFIG)
        self.assertIsNotNone(se)
        self.assertEqual(se.num_entities, 5)
        self.assertEqual(se.num_relations, 5)
        self.assertEqual(se.embedding_dim, 5)
        self.assertEqual(se.l_p_norm_entities, 2)
        self.assertEqual(se.margin_loss, 4)

    def test_instantiate_unstructured_model(self):
        """Test that UnstructuredModel can be instantiated."""
        um = UnstructuredModel(config=UM_CONFIG)
        self.assertIsNotNone(um)
        self.assertEqual(um.num_entities, 5)
        self.assertEqual(um.num_relations, 5)
        self.assertEqual(um.embedding_dim, 5)
        self.assertEqual(um.l_p_norm_entities, 2)
        self.assertEqual(um.margin_loss, 4)

    def test_instantiate_rescal(self):
        """Test that RESCAL can be instantiated."""
        rescal = RESCAL(config=RESCAL_CONFIG)
        self.assertIsNotNone(rescal)
        self.assertEqual(rescal.num_entities, 5)
        self.assertEqual(rescal.num_relations, 5)
        self.assertEqual(rescal.embedding_dim, 5)
        self.assertEqual(rescal.margin_loss, 4)

    def test_instantiate_conv_e(self):
        """Test that ConvE can be instantiated."""
        conv_e = ConvE(config=CONV_E_CONFIG)
        self.assertIsNotNone(conv_e)
        self.assertEqual(conv_e.num_entities, 5)
        self.assertEqual(conv_e.num_relations, 5)
        self.assertEqual(conv_e.embedding_dim, 5)


class TestScoringFunctions(unittest.TestCase):
    def test_compute_scores_trans_e(self):
        """Test that TransE's socore function computes the scores correct."""
        trans_e = TransE(config=TRANS_E_CONFIG)
        h_embs = torch.tensor([[1.,1.],[2.,2.]],dtype=torch.float)
        r_embs = torch.tensor([[1., 1.], [2., 2.]],dtype=torch.float)
        t_embs = torch.tensor([[2., 2.], [4., 4.]],dtype=torch.float)

        scores = trans_e._compute_scores(h_embs,r_embs,t_embs).cpu().numpy().tolist()

        self.assertEqual(scores, [0.,0.])

    def test_compute_scores_trans_h(self):
        """Test that TransH's socore function computes the scores correct."""
        trans_h = TransH(config=TRANS_H_CONFIG)
        proj_h_embs = torch.tensor([[1.,1.],[1.,1.]],dtype=torch.float)
        proj_r_embs = torch.tensor([[1., 1.], [2., 2.]],dtype=torch.float)
        proj_t_embs = torch.tensor([[2., 2.], [4., 4.]],dtype=torch.float)

        scores = trans_h._compute_scores(proj_h_embs,proj_r_embs,proj_t_embs).cpu().numpy().tolist()

        self.assertEqual(scores, [0.,4.])

    def test_compute_scores_trans_r(self):
        """Test that TransR's socore function computes the scores correct."""
        trans_r = TransR(config=TRANS_R_CONFIG)
        proj_h_embs = torch.tensor([[1.,1.],[1.,1.]],dtype=torch.float)
        proj_r_embs = torch.tensor([[1., 1.], [2., 2.]],dtype=torch.float)
        proj_t_embs = torch.tensor([[2., 2.], [4., 4.]],dtype=torch.float)

        scores = trans_r._compute_scores(proj_h_embs,proj_r_embs,proj_t_embs).cpu().numpy().tolist()

        self.assertEqual(scores, [0.,4.])

    def test_compute_scores_trans_d(self):
        """Test that TransD's socore function computes the scores correct."""
        trans_d = TransD(config=TRANS_D_CONFIG)
        proj_h_embs = torch.tensor([[1.,1.],[1.,1.]],dtype=torch.float)
        proj_r_embs = torch.tensor([[1., 1.], [2., 2.]],dtype=torch.float)
        proj_t_embs = torch.tensor([[2., 2.], [4., 4.]],dtype=torch.float)

        scores = trans_d._compute_scores(proj_h_embs,proj_r_embs,proj_t_embs).cpu().numpy().tolist()

        self.assertEqual(scores, [0.,4.])

    def test_compute_scores_distmult(self):
        """Test that DistMult's socore function computes the scores correct."""
        distmult = DistMult(config=DISTMULT_CONFIG)
        h_embs = torch.tensor([[1.,1.],[1.,1.]],dtype=torch.float)
        r_embs = torch.tensor([[1., 1.], [2., 2.]],dtype=torch.float)
        t_embs = torch.tensor([[2., 2.], [4., 4.]],dtype=torch.float)

        scores = distmult._compute_scores(h_embs,r_embs,t_embs).cpu().numpy().tolist()

        self.assertEqual(scores, [-4.,-16.])

    def test_compute_scores_um(self):
        """Test that DistMult's socore function computes the scores correct."""
        um = UnstructuredModel(config=UM_CONFIG)
        h_embs = torch.tensor([[1.,1.],[1.,1.]],dtype=torch.float)
        t_embs = torch.tensor([[2., 2.], [4., 4.]],dtype=torch.float)

        scores = um._compute_scores(h_embs, t_embs).cpu().numpy().tolist()

        self.assertEqual(scores, [4.,36.])

    def test_compute_scores_se(self):
        """Test that SE's socore function computes the scores correct."""
        se = StructuredEmbedding(config=SE_CONFIG)
        proj_h_embs = torch.tensor([[1.,1.],[1.,1.]],dtype=torch.float)
        proj_t_embs = torch.tensor([[2., 2.], [4., 4.]],dtype=torch.float)

        scores = se._compute_scores(proj_h_embs,proj_t_embs).cpu().numpy().tolist()

        self.assertEqual(scores, [2.,6.])

    def test_compute_scores_ermlp(self):
        """Test that SE's score function computes the scores correct."""
        ermlp = ERMLP(config=ERMLP_CONFIG)

        h_embs = torch.tensor([[1., 1.], [1., 1.]], dtype=torch.float)
        r_embs = torch.tensor([[1., 1.], [2., 2.]], dtype=torch.float)
        t_embs = torch.tensor([[2., 2.], [4., 4.]], dtype=torch.float)

        scores = ermlp._compute_scores(h_embs,r_embs,t_embs).detach().cpu().numpy().tolist()

        self.assertEqual(len(scores),2)


    def test_um_predict(self):
        """Test UM's predict function."""
        um = UnstructuredModel(config=UM_CONFIG)
        predictions = um.predict(triples=TEST_TRIPLES)

        self.assertEqual(len(predictions),len(TEST_TRIPLES))
        self.assertTrue(type(predictions.shape[0]),float)

    def test_se_predict(self):
        """Test SE's predict function."""
        se = StructuredEmbedding(config=SE_CONFIG)
        predictions = se.predict(triples=TEST_TRIPLES)

        self.assertEqual(len(predictions),len(TEST_TRIPLES))
        self.assertTrue(type(predictions.shape[0]),float)

    def test_trans_e_predict(self):
        """Test TransE's predict function."""
        trans_e = TransE(config=TRANS_E_CONFIG)
        predictions = trans_e.predict(triples=TEST_TRIPLES)

        self.assertEqual(len(predictions),len(TEST_TRIPLES))
        self.assertTrue(type(predictions.shape[0]),float)

    def test_trans_h_predict(self):
        """Test TransH's predict function."""
        trans_h = TransE(config=TRANS_H_CONFIG)
        predictions = trans_h.predict(triples=TEST_TRIPLES)

        self.assertEqual(len(predictions),len(TEST_TRIPLES))
        self.assertTrue(type(predictions.shape[0]),float)

    def test_trans_r_predict(self):
        """Test TransR's predict function."""
        trans_r = TransR(config=TRANS_R_CONFIG)
        predictions = trans_r.predict(triples=TEST_TRIPLES)

        self.assertEqual(len(predictions),len(TEST_TRIPLES))
        self.assertTrue(type(predictions.shape[0]),float)

    def test_trans_d_predict(self):
        """Test TransD's predict function."""
        trans_d = TransR(config=TRANS_D_CONFIG)
        predictions = trans_d.predict(triples=TEST_TRIPLES)

        self.assertEqual(len(predictions),len(TEST_TRIPLES))
        self.assertTrue(type(predictions.shape[0]),float)

    def test_ermlp_predict(self):
        """Test ERMLP's predict function."""
        ermlp = ERMLP(config=ERMLP_CONFIG)
        predictions = ermlp.predict(triples=TEST_TRIPLES)

        self.assertEqual(len(predictions),len(TEST_TRIPLES))
        self.assertTrue(type(predictions.shape[0]),float)

    def test_rescal_predict(self):
        """Test RESCAL's predict function."""
        rescal = RESCAL(config=RESCAL_CONFIG)
        predictions = rescal.predict(triples=TEST_TRIPLES)

        self.assertEqual(len(predictions),len(TEST_TRIPLES))
        self.assertTrue(type(predictions.shape[0]),float)

    def test_conv_e_predict(self):
        """Test ConvE's predict function."""
        conv_e = ConvE(config=CONV_E_CONFIG)

        predictions = conv_e.predict(triples=TEST_TRIPLES)

        self.assertEqual(len(predictions), len(TEST_TRIPLES))
        self.assertTrue(type(predictions.shape[0]),int)



