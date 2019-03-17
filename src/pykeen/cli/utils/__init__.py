# -*- coding: utf-8 -*-

"""Utilities for the command line interface."""

from .conv_e_cli import configure_conv_e_hpo_pipeline, configure_conv_e_training_pipeline
from .distmult_cli import configure_distmult_hpo_pipeline, configure_distmult_training_pipeline
from .ermlp_cli import configure_ermlp_hpo_pipeline, configure_ermlp_training_pipeline
from .rescal_cli import configure_rescal_hpo_pipeline, configure_rescal_training_pipeline
from .structured_embedding_cli import configure_se_hpo_pipeline, configure_se_training_pipeline
from .trans_d_cli import configure_trans_d_hpo_pipeline, configure_trans_d_training_pipeline
from .trans_e_cli import configure_trans_e_hpo_pipeline, configure_trans_e_training_pipeline
from .trans_h_cli import configure_trans_h_hpo_pipeline, configure_trans_h_training_pipeline
from .trans_r_cli import configure_trans_r_hpo_pipeline, configure_trans_r_training_pipeline
from .unstructured_model_cli import configure_um_hpo_pipeline, configure_um_training_pipeline

__all__ = [
    'configure_conv_e_hpo_pipeline', 'configure_conv_e_training_pipeline',
    'configure_distmult_hpo_pipeline', 'configure_distmult_training_pipeline',
    'configure_ermlp_hpo_pipeline', 'configure_ermlp_training_pipeline',
    'configure_rescal_hpo_pipeline', 'configure_rescal_training_pipeline',
    'configure_se_hpo_pipeline', 'configure_se_training_pipeline',
    'configure_trans_d_hpo_pipeline', 'configure_trans_d_training_pipeline',
    'configure_trans_e_hpo_pipeline', 'configure_trans_e_training_pipeline',
    'configure_trans_h_hpo_pipeline', 'configure_trans_h_training_pipeline',
    'configure_trans_r_hpo_pipeline', 'configure_trans_r_training_pipeline',
    'configure_um_hpo_pipeline', 'configure_um_training_pipeline',
]
