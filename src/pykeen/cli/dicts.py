# -*- coding: utf-8 -*-

"""PyKEEN's command line interface."""

from pykeen.constants import (
    CONV_E_NAME, DISTMULT_NAME, ERMLP_NAME, RESCAL_NAME, SE_NAME, TRANS_D_NAME, TRANS_E_NAME, TRANS_H_NAME,
    TRANS_R_NAME, UM_NAME,
)
from pykeen.cli.utils import (
    configure_conv_e_hpo_pipeline, configure_conv_e_training_pipeline,
    configure_distmult_hpo_pipeline, configure_distmult_training_pipeline, configure_ermlp_hpo_pipeline,
    configure_ermlp_training_pipeline, configure_rescal_hpo_pipeline, configure_rescal_training_pipeline,
    configure_se_hpo_pipeline, configure_se_training_pipeline, configure_trans_d_hpo_pipeline,
    configure_trans_d_training_pipeline, configure_trans_e_hpo_pipeline, configure_trans_e_training_pipeline,
    configure_trans_h_hpo_pipeline, configure_trans_h_training_pipeline, configure_trans_r_hpo_pipeline,
    configure_trans_r_training_pipeline, configure_um_hpo_pipeline, configure_um_training_pipeline,
)

__all__ = [
    'MODEL_TRAINING_CONFIG_FUNCS',
    'MODEL_HPO_CONFIG_FUNCS',
]

MODEL_TRAINING_CONFIG_FUNCS = {
    TRANS_E_NAME: configure_trans_e_training_pipeline,
    TRANS_H_NAME: configure_trans_h_training_pipeline,
    TRANS_R_NAME: configure_trans_r_training_pipeline,
    TRANS_D_NAME: configure_trans_d_training_pipeline,
    SE_NAME: configure_se_training_pipeline,
    UM_NAME: configure_um_training_pipeline,
    DISTMULT_NAME: configure_distmult_training_pipeline,
    ERMLP_NAME: configure_ermlp_training_pipeline,
    RESCAL_NAME: configure_rescal_training_pipeline,
    CONV_E_NAME: configure_conv_e_training_pipeline
}

MODEL_HPO_CONFIG_FUNCS = {
    TRANS_E_NAME: configure_trans_e_hpo_pipeline,
    TRANS_H_NAME: configure_trans_h_hpo_pipeline,
    TRANS_R_NAME: configure_trans_r_hpo_pipeline,
    TRANS_D_NAME: configure_trans_d_hpo_pipeline,
    SE_NAME: configure_se_hpo_pipeline,
    UM_NAME: configure_um_hpo_pipeline,
    DISTMULT_NAME: configure_distmult_hpo_pipeline,
    ERMLP_NAME: configure_ermlp_hpo_pipeline,
    RESCAL_NAME: configure_rescal_hpo_pipeline,
    CONV_E_NAME: configure_conv_e_hpo_pipeline
}
