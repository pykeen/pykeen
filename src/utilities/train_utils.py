import logging
import timeit

import numpy as np
import torch
import torch.optim as optim

from utilities.constants import CONV_E, TRANS_E, TRANS_H, TRANS_D, TRANS_R, ROT_E

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def split_list_in_batches(input_list, batch_size):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]


def train_model(kg_embedding_model, all_entities, learning_rate, num_epochs, batch_size, pos_triples, device, seed):
    model_name = kg_embedding_model.model_name

    if model_name in [TRANS_E, TRANS_H, TRANS_D, TRANS_R, ROT_E]:
        return train_trans_x_model(kg_embedding_model, all_entities, learning_rate, num_epochs, batch_size, pos_triples,
                                   device, seed)

    if model_name == CONV_E:
        return train_conv_e_model(kg_embedding_model, learning_rate, num_epochs, batch_size, pos_triples, device, seed)


def train_trans_x_model(kg_embedding_model, all_entities, learning_rate, num_epochs, batch_size, pos_triples, device,
                        seed):
    kg_embedding_model = kg_embedding_model.to(device)

    optimizer = optim.SGD(kg_embedding_model.parameters(), lr=learning_rate)

    loss_per_epoch = []

    log.info('****Run Model On %s****' % str(device).upper())

    num_pos_triples = pos_triples.shape[0]
    num_entities = all_entities.shape[0]

    for epoch in range(num_epochs):
        np.random.seed(seed=seed)
        indices = np.arange(num_pos_triples)
        np.random.shuffle(indices)
        pos_triples = pos_triples[indices]
        start = timeit.default_timer()
        pos_batches = split_list_in_batches(input_list=pos_triples, batch_size=batch_size)
        current_epoch_loss = 0.

        for i in range(len(pos_batches)):
            # index_of_batch = np.random.choice(np.arange(0,len(pos_batches)))
            pos_batch = pos_batches[i]
            current_batch_size = len(pos_batch)
            batch_subjs = pos_batch[:, 0:1]
            batch_preds = pos_batch[:, 1:2]
            batch_objs = pos_batch[:, 2:3]

            num_subj_corrupt = len(pos_batch) // 2
            num_obj_corrupt = len(pos_batch) - num_subj_corrupt
            pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=device)

            corrupted_subj_indices = np.random.choice(np.arange(0, num_entities), size=num_subj_corrupt)
            corrupted_subjects = np.reshape(all_entities[corrupted_subj_indices], newshape=(-1, 1))
            subject_based_corrupted_triples = np.concatenate(
                [corrupted_subjects, batch_preds[:num_subj_corrupt], batch_objs[:num_subj_corrupt]], axis=1)

            corrupted_obj_indices = np.random.choice(np.arange(0, num_entities), size=num_obj_corrupt)
            corrupted_objects = np.reshape(all_entities[corrupted_obj_indices], newshape=(-1, 1))

            object_based_corrupted_triples = np.concatenate(
                [batch_subjs[num_subj_corrupt:], batch_preds[num_subj_corrupt:], corrupted_objects], axis=1)

            neg_batch = np.concatenate([subject_based_corrupted_triples, object_based_corrupted_triples], axis=0)

            neg_batch = torch.tensor(neg_batch, dtype=torch.long, device=device)

            # Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            optimizer.zero_grad()
            loss = kg_embedding_model(pos_batch, neg_batch)
            current_epoch_loss += (loss.item() * current_batch_size)

            loss.backward()
            optimizer.step()

        stop = timeit.default_timer()
        log.info("Epoch %s took %s seconds \n" % (str(epoch), str(round(stop - start))))
        # Track epoch loss
        loss_per_epoch.append(current_epoch_loss / len(pos_triples))

    return kg_embedding_model, loss_per_epoch


def train_conv_e_model(kg_embedding_model, learning_rate, num_epochs, batch_size, pos_triples, device, seed):
    np.random.seed(seed=seed)
    indices = np.arange(pos_triples.shape[0])
    np.random.shuffle(indices)
    pos_triples = pos_triples[indices]
    num_pos_triples = pos_triples.shape[0]

    # Create labels
    subject_relation_pairs = pos_triples[:, 0:2]
    entities = np.arange(kg_embedding_model.num_entities)
    labels = []

    for subj_rel in subject_relation_pairs:
        subj_rel_rep = np.repeat(subj_rel,axis=0)
        label = (pos_triples[:,0:2] == subj_rel).all(axis=1)
        # objects = pos_triples[mat, 2:3]
        # objects = np.unique(np.ndarray.flatten(objects))
        # label_vec = np.in1d(entities, objects) * 1
        # labels.append(label_vec)

    kg_embedding_model = kg_embedding_model.to(device)
    optimizer = optim.SGD(kg_embedding_model.parameters(), lr=learning_rate)
    total_loss = 0
    loss_per_epoch = []

    log.info('****Run Model On %s****' % str(device).upper())
    # Train
    for epoch in range(num_epochs):
        np.random.seed(seed=seed)
        indices = np.arange(num_pos_triples)
        np.random.shuffle(indices)
        pos_triples = pos_triples[indices]
        subject_relation_pairs = pos_triples[:, 0:2]
        start = timeit.default_timer()
        pos_batches = split_list_in_batches(input_list=subject_relation_pairs, batch_size=batch_size)
        # label_batches = split_list_in_batches(input_list=labels, batch_size=batch_size)
        current_epoch_loss = 0.

        for i in range(len(pos_batches)):
            optimizer.zero_grad()
            pos_batch = pos_batches[i]
            # label_batch = label_batches[i]
            pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=device)
            # label_batch = torch.tensor(label_batch, dtype=torch.float, device=device)

            predictions = kg_embedding_model(pos_batch[:, 0:1], pos_batch[:, 1:2])
            loss = kg_embedding_model.compute_loss(pred=predictions, targets=label_batch)
            loss.backward()
            optimizer.step()
            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
            current_epoch_loss += loss.item()
        stop = timeit.default_timer()
        log.info("Epoch %s took %s seconds \n" % (str(epoch), str(round(stop - start))))
        loss_per_epoch.append(current_epoch_loss)

    return kg_embedding_model, loss_per_epoch

