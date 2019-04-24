# -*- coding: utf-8 -*-

"""Utilities for training KGE models."""

import logging
import timeit
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from sklearn.utils import shuffle
from torch.nn import Module
from tqdm import trange

import pykeen.constants as pkc
from pykeen.kge_models import ConvE

__all__ = [
    'train_kge_model',
]

log = logging.getLogger(__name__)


def _split_list_in_batches(input_list, batch_size):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]


def train_kge_model(
        kge_model: Module,
        all_entities,
        learning_rate,
        num_epochs,
        batch_size,
        pos_triples,
        device,
        seed: Optional[int] = None,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[Module, List[float]]:
    """Train the model."""
    if pkc.CONV_E_NAME == kge_model.model_name:
        return _train_conv_e_model(
            kge_model=kge_model,
            all_entities=all_entities,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            pos_triples=pos_triples,
            device=device,
            seed=seed,
            tqdm_kwargs=tqdm_kwargs
        )

    # model_name in {TRANS_E_NAME, TRANS_H_NAME, TRANS_D_NAME, TRANS_R_NAME, DISTMULT_NAME, UM_NAME, SE_NAME, ERMLP_NAME, RESCAL_NAME}
    return _train_basic_model(
        kge_model=kge_model,
        all_entities=all_entities,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        pos_triples=pos_triples,
        device=device,
        seed=seed,
        tqdm_kwargs=tqdm_kwargs,
    )


def _train_basic_model(
        kge_model: Module,
        all_entities,
        learning_rate,
        num_epochs,
        batch_size,
        pos_triples,
        device,
        seed: Optional[int] = None,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[Module, List[float]]:
    """"""
    if seed is not None:
        np.random.seed(seed=seed)

    kge_model = kge_model.to(device)

    optimizer = optim.SGD(kge_model.parameters(), lr=learning_rate)
    log.debug(f'****running model on {device}****')

    loss_per_epoch = []
    num_pos_triples = pos_triples.shape[0]
    num_entities = all_entities.shape[0]

    start_training = timeit.default_timer()

    _tqdm_kwargs = dict(desc='Training epoch')
    if tqdm_kwargs:
        _tqdm_kwargs.update(tqdm_kwargs)

    for _ in trange(num_epochs, **_tqdm_kwargs):
        indices = np.arange(num_pos_triples)
        np.random.shuffle(indices)
        pos_triples = pos_triples[indices]
        pos_batches = _split_list_in_batches(input_list=pos_triples, batch_size=batch_size)
        current_epoch_loss = 0.

        for i, pos_batch in enumerate(pos_batches):
            current_batch_size = len(pos_batch)
            batch_subjs = pos_batch[:, 0:1]
            batch_relations = pos_batch[:, 1:2]
            batch_objs = pos_batch[:, 2:3]

            num_subj_corrupt = len(pos_batch) // 2
            num_obj_corrupt = len(pos_batch) - num_subj_corrupt
            pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=device)

            corrupted_subj_indices = np.random.choice(np.arange(0, num_entities), size=num_subj_corrupt)
            corrupted_subjects = np.reshape(all_entities[corrupted_subj_indices], newshape=(-1, 1))
            subject_based_corrupted_triples = np.concatenate(
                [corrupted_subjects, batch_relations[:num_subj_corrupt], batch_objs[:num_subj_corrupt]], axis=1)

            corrupted_obj_indices = np.random.choice(np.arange(0, num_entities), size=num_obj_corrupt)
            corrupted_objects = np.reshape(all_entities[corrupted_obj_indices], newshape=(-1, 1))

            object_based_corrupted_triples = np.concatenate(
                [batch_subjs[num_subj_corrupt:], batch_relations[num_subj_corrupt:], corrupted_objects], axis=1)

            neg_batch = np.concatenate([subject_based_corrupted_triples, object_based_corrupted_triples], axis=0)

            neg_batch = torch.tensor(neg_batch, dtype=torch.long, device=device)

            # Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old instance
            optimizer.zero_grad()
            loss = kge_model(pos_batch, neg_batch)
            current_epoch_loss += (loss.item() * current_batch_size)

            loss.backward()
            optimizer.step()

        # log.info("Epoch %s took %s seconds \n" % (str(epoch), str(round(stop - start))))
        # Track epoch loss
        loss_per_epoch.append(current_epoch_loss / len(pos_triples))

    stop_training = timeit.default_timer()
    log.debug("training took %.2fs seconds", stop_training - start_training)

    return kge_model, loss_per_epoch


def _train_conv_e_model(
        kge_model: ConvE,
        all_entities,
        learning_rate,
        num_epochs,
        batch_size,
        pos_triples, device,
        seed: Optional[int] = None,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[ConvE, List[float]]:
    """"""
    if seed is not None:
        np.random.seed(seed=seed)

    kge_model = kge_model.to(device)

    optimizer = optim.Adam(kge_model.parameters(), lr=learning_rate)

    loss_per_epoch: List[float] = []

    log.info('****Run Model On %s****' % str(device).upper())

    num_pos_triples = pos_triples.shape[0]
    num_entities = all_entities.shape[0]

    start_training = timeit.default_timer()

    _tqdm_kwargs = dict(desc='Training epoch')
    if tqdm_kwargs:
        _tqdm_kwargs.update(tqdm_kwargs)

    for epoch in trange(num_epochs, **_tqdm_kwargs):
        indices = np.arange(num_pos_triples)
        np.random.shuffle(indices)
        pos_triples = pos_triples[indices]
        num_positives = batch_size // 2
        # TODO: Make sure that batch = num_pos + num_negs
        # num_negatives = batch_size - num_positives

        pos_batches = _split_list_in_batches(input_list=pos_triples, batch_size=num_positives)
        current_epoch_loss = 0.

        for i in range(len(pos_batches)):
            # TODO: Remove original subject and object from entity set
            pos_batch = pos_batches[i]
            current_batch_size = len(pos_batch)
            batch_subjs = pos_batch[:, 0:1]
            batch_relations = pos_batch[:, 1:2]
            batch_objs = pos_batch[:, 2:3]

            num_subj_corrupt = len(pos_batch) // 2
            num_obj_corrupt = len(pos_batch) - num_subj_corrupt
            pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=device)

            corrupted_subj_indices = np.random.choice(np.arange(0, num_entities), size=num_subj_corrupt)
            corrupted_subjects = np.reshape(all_entities[corrupted_subj_indices], newshape=(-1, 1))
            subject_based_corrupted_triples = np.concatenate(
                [corrupted_subjects, batch_relations[:num_subj_corrupt], batch_objs[:num_subj_corrupt]], axis=1)

            corrupted_obj_indices = np.random.choice(np.arange(0, num_entities), size=num_obj_corrupt)
            corrupted_objects = np.reshape(all_entities[corrupted_obj_indices], newshape=(-1, 1))

            object_based_corrupted_triples = np.concatenate(
                [batch_subjs[num_subj_corrupt:], batch_relations[num_subj_corrupt:], corrupted_objects], axis=1)

            neg_batch = np.concatenate([subject_based_corrupted_triples, object_based_corrupted_triples], axis=0)

            neg_batch = torch.tensor(neg_batch, dtype=torch.long, device=device)

            batch = np.concatenate([pos_batch, neg_batch], axis=0)
            positive_labels = np.ones(shape=current_batch_size)
            negative_labels = np.zeros(shape=current_batch_size)
            labels = np.concatenate([positive_labels, negative_labels], axis=0)

            batch, labels = shuffle(batch, labels, random_state=seed)

            batch = torch.tensor(batch, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.float)

            # Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            optimizer.zero_grad()
            loss = kge_model(batch, labels)
            current_epoch_loss += (loss.item() * current_batch_size)

            loss.backward()
            optimizer.step()

        # log.info("Epoch %s took %s seconds \n" % (str(epoch), str(round(stop - start))))
        # Track epoch loss
        loss_per_epoch.append(current_epoch_loss / len(pos_triples))

    stop_training = timeit.default_timer()
    log.info("Training took %s seconds \n" % (str(round(stop_training - start_training))))

    return kge_model, loss_per_epoch
