# -*- coding: utf-8 -*-

"""Test that all models can be instantiated."""

import unittest

from pykeen.constants import *
from pykeen.kge_models import TransE, TransH, TransR, TransD, DistMult, ERMLP, StructuredEmbedding, UnstructuredModel, \
    RESCAL, ConvE


class TestModelInstantiation(unittest.TestCase):
    """Test that all models can be instantiated."""

    def test_instantiate_trans_e(self):
        """Test that TransE can be instantiated."""
        trans_e = TransE(config={
            NUM_ENTITIES: 5,
            NUM_RELATIONS: 5,
            EMBEDDING_DIM: 5,
            NORM_FOR_NORMALIZATION_OF_ENTITIES: 2,
            SCORING_FUNCTION_NORM: 1,
            MARGIN_LOSS: 4,
        })
        self.assertIsNotNone(trans_e)
        self.assertTrue(trans_e.num_entities, 5)
        self.assertTrue(trans_e.num_relations, 5)
        self.assertTrue(trans_e.embedding_dim, 5)
        self.assertTrue(trans_e.l_p_norm_entities, 2)
        self.assertTrue(trans_e.scoring_fct_norm, 1)
        self.assertTrue(trans_e.margin_loss, 4)

    def test_instantiate_trans_h(self):
        """Test that TransH can be instantiated."""
        trans_h = TransH(config={
            NUM_ENTITIES: 5,
            NUM_RELATIONS: 5,
            EMBEDDING_DIM: 5,
            NORM_FOR_NORMALIZATION_OF_ENTITIES: 2,
            SCORING_FUNCTION_NORM: 1,
            MARGIN_LOSS: 4,
            WEIGHT_SOFT_CONSTRAINT_TRANS_H: 0.05,
        })
        self.assertIsNotNone(trans_h)
        self.assertTrue(trans_h.num_entities,5)
        self.assertTrue(trans_h.num_relations, 5)
        self.assertTrue(trans_h.embedding_dim, 5)
        self.assertTrue(trans_h.weightning_soft_constraint,0.05)
        self.assertTrue(trans_h.scoring_fct_norm, 1)
        self.assertTrue(trans_h.margin_loss, 4)

    def test_instantiate_trans_r(self):
        """Test that TransR can be instantiated."""
        trans_r = TransR(config={
            NUM_ENTITIES: 5,
            NUM_RELATIONS: 5,
            EMBEDDING_DIM: 5,
            RELATION_EMBEDDING_DIM: 3,
            SCORING_FUNCTION_NORM: 1,
            MARGIN_LOSS: 4,
        })
        self.assertIsNotNone(trans_r)
        self.assertTrue(trans_r.num_entities,5)
        self.assertTrue(trans_r.num_relations, 5)
        self.assertTrue(trans_r.embedding_dim, 5)
        self.assertTrue(trans_r.relation_embedding_dim, 3)
        self.assertTrue(trans_r.scoring_fct_norm, 1)
        self.assertTrue(trans_r.margin_loss, 4)

    def test_instantiate_trans_d(self):
        """Test that TransD can be instantiated."""
        trans_d = TransD(config={
            NUM_ENTITIES: 5,
            NUM_RELATIONS: 5,
            EMBEDDING_DIM: 5,
            RELATION_EMBEDDING_DIM: 3,
            SCORING_FUNCTION_NORM: 1,
            MARGIN_LOSS: 4,
        })
        self.assertIsNotNone(trans_d)
        self.assertTrue(trans_d.num_entities,5)
        self.assertTrue(trans_d.num_relations, 5)
        self.assertTrue(trans_d.embedding_dim, 5)
        self.assertTrue(trans_d.relation_embedding_dim, 3)
        self.assertTrue(trans_d.scoring_fct_norm, 1)
        self.assertTrue(trans_d.margin_loss, 4)

    def test_instantiate_distmult(self):
        """Test that DistMult can be instantiated."""
        distmult = DistMult(config={
            NUM_ENTITIES: 5,
            NUM_RELATIONS: 5,
            EMBEDDING_DIM: 5,
            SCORING_FUNCTION_NORM: 1,
            MARGIN_LOSS: 4,
        })
        self.assertIsNotNone(distmult)
        self.assertTrue(distmult.num_entities,5)
        self.assertTrue(distmult.num_relations, 5)
        self.assertTrue(distmult.embedding_dim, 5)
        self.assertTrue(distmult.margin_loss, 4)

    def test_instantiate_ermlp(self):
        """Test that ERMLP can be instantiated."""
        ermlp = ERMLP(config={
            NUM_ENTITIES: 5,
            NUM_RELATIONS: 5,
            EMBEDDING_DIM: 5,
            SCORING_FUNCTION_NORM: 1,
            MARGIN_LOSS: 4,
        })
        self.assertIsNotNone(ermlp)
        self.assertTrue(ermlp.num_entities,5)
        self.assertTrue(ermlp.num_relations, 5)
        self.assertTrue(ermlp.embedding_dim, 5)
        self.assertTrue(ermlp.margin_loss, 4)

    def test_instantiate_strcutured_embedding(self):
        """Test that StructuredEmbedding can be instantiated."""
        se = StructuredEmbedding(config={
            NUM_ENTITIES: 5,
            NUM_RELATIONS: 5,
            EMBEDDING_DIM: 5,
            NORM_FOR_NORMALIZATION_OF_ENTITIES: 2,
            SCORING_FUNCTION_NORM: 1,
            MARGIN_LOSS: 4,
        })
        self.assertIsNotNone(se)
        self.assertTrue(se.num_entities,5)
        self.assertTrue(se.num_relations, 5)
        self.assertTrue(se.embedding_dim, 5)
        self.assertTrue(se.l_p_norm_entities, 2)
        self.assertTrue(se.margin_loss, 4)

    def test_instantiate_unstructured_model(self):
        """Test that UnstructuredModel can be instantiated."""
        um = UnstructuredModel(config={
            NUM_ENTITIES: 5,
            NUM_RELATIONS: 5,
            EMBEDDING_DIM: 5,
            NORM_FOR_NORMALIZATION_OF_ENTITIES: 2,
            SCORING_FUNCTION_NORM: 1,
            MARGIN_LOSS: 4,
        })
        self.assertIsNotNone(um)
        self.assertTrue(um.num_entities, 5)
        self.assertTrue(um.num_relations, 5)
        self.assertTrue(um.embedding_dim, 5)
        self.assertTrue(um.l_p_norm_entities, 2)
        self.assertTrue(um.margin_loss, 4)

    def test_instantiate_rescal(self):
        """Test that RESCAL can be instantiated."""
        rescal = RESCAL(config={
            NUM_ENTITIES: 5,
            NUM_RELATIONS: 5,
            EMBEDDING_DIM: 5,
            NORM_FOR_NORMALIZATION_OF_ENTITIES: 2,
            SCORING_FUNCTION_NORM: 1,
            MARGIN_LOSS: 4,
        })
        self.assertIsNotNone(rescal)
        self.assertTrue(rescal.num_entities, 5)
        self.assertTrue(rescal.num_relations, 5)
        self.assertTrue(rescal.embedding_dim, 5)
        self.assertTrue(rescal.margin_loss, 4)

    def test_instantiate_conv_e(self):
        """Test that ConvE can be instantiated."""
        conv_e = ConvE(config={
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
        })
        self.assertIsNotNone(conv_e)
        self.assertTrue(conv_e.num_entities, 5)
        self.assertTrue(conv_e.num_relations, 5)
        self.assertTrue(conv_e.embedding_dim, 5)