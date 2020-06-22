# -*- coding: utf-8 -*-

"""PyKEEN's command line interface."""

from pykeen.cli.utils import (
    configure_conv_e_hpo_pipeline, configure_conv_e_training_pipeline,
    configure_distmult_hpo_pipeline, configure_distmult_training_pipeline, configure_ermlp_hpo_pipeline,
    configure_ermlp_training_pipeline, configure_rescal_hpo_pipeline, configure_rescal_training_pipeline,
    configure_se_hpo_pipeline, configure_se_training_pipeline, configure_trans_d_hpo_pipeline,
    configure_trans_d_training_pipeline, configure_trans_e_hpo_pipeline, configure_trans_e_training_pipeline,
    configure_trans_h_hpo_pipeline, configure_trans_h_training_pipeline, configure_trans_r_hpo_pipeline,
    configure_trans_r_training_pipeline, configure_um_hpo_pipeline, configure_um_training_pipeline,
)
from pykeen.kge_models import (
    ConvE, DistMult, ERMLP, RESCAL, StructuredEmbedding, TransD, TransE, TransH, TransR, UnstructuredModel,
)

__all__ = [
    'MODEL_TRAINING_CONFIG_FUNCS',
    'MODEL_HPO_CONFIG_FUNCS',
]

MODEL_TRAINING_CONFIG_FUNCS = {
    TransE.model_name: configure_trans_e_training_pipeline,
    TransH.model_name: configure_trans_h_training_pipeline,
    TransR.model_name: configure_trans_r_training_pipeline,
    TransD.model_name: configure_trans_d_training_pipeline,
    StructuredEmbedding.model_name: configure_se_training_pipeline,
    UnstructuredModel.model_name: configure_um_training_pipeline,
    DistMult.model_name: configure_distmult_training_pipeline,
    ERMLP.model_name: configure_ermlp_training_pipeline,
    RESCAL.model_name: configure_rescal_training_pipeline,
    ConvE.model_name: configure_conv_e_training_pipeline,
}

MODEL_HPO_CONFIG_FUNCS = {
    TransE.model_name: configure_trans_e_hpo_pipeline,
    TransH.model_name: configure_trans_h_hpo_pipeline,
    TransR.model_name: configure_trans_r_hpo_pipeline,
    TransD.model_name: configure_trans_d_hpo_pipeline,
    StructuredEmbedding.model_name: configure_se_hpo_pipeline,
    UnstructuredModel.model_name: configure_um_hpo_pipeline,
    DistMult.model_name: configure_distmult_hpo_pipeline,
    ERMLP.model_name: configure_ermlp_hpo_pipeline,
    RESCAL.model_name: configure_rescal_hpo_pipeline,
    ConvE.model_name: configure_conv_e_hpo_pipeline,
}
