import logging
import timeit

import numpy as np
import torch
import torch.optim as optim

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def split_list_in_batches(input_list, batch_size):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]


def train(kg_embedding_model, learning_rate, num_epochs, batch_size, pos_triples, device, seed):
    np.random.seed(seed=seed)
    indices = np.arange(pos_triples.shape[0])
    np.random.shuffle(indices)
    pos_triples = pos_triples[indices]
    kg_embedding_model = kg_embedding_model.to(device)

    optimizer = optim.SGD(kg_embedding_model.parameters(), lr=learning_rate)

    total_loss = 0

    log.info('****Run Model On %s****' % str(device).upper())

    subjects = pos_triples[:, 0:1]
    objects = pos_triples[:, 2:3]

    for epoch in range(num_epochs):
        start = timeit.default_timer()
        pos_batches = split_list_in_batches(input_list=pos_triples, batch_size=batch_size)
        # neg_batches = split_list_in_batches(input_list=neg_triples, batch_size=batch_size)
        for i in range(len(pos_batches)):
            pos_batch = pos_batches[i]
            batch_subjs = pos_batch[:, 0:1]
            batch_preds = pos_batch[:, 1:2]
            batch_objs = pos_batch[:, 2:3]

            num_subj_corrupt = len(pos_batch) // 2
            num_obj_corrupt = len(pos_batch) - num_subj_corrupt
            pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=device)

            corrupted_subj_indices = np.random.choice(np.arange(0, len(pos_batch)), size=num_subj_corrupt)
            corrupted_subjects = subjects[corrupted_subj_indices]

            subject_based_corrupted_triples = np.concatenate(
                [corrupted_subjects, batch_preds[:num_subj_corrupt], batch_objs[:num_subj_corrupt]], axis=1)

            corrupted_obj_indices = np.random.choice(np.arange(0, len(pos_batch)), size=num_obj_corrupt)
            corrupted_objects = objects[corrupted_obj_indices]

            object_based_corrupted_triples = np.concatenate(
                [batch_subjs[num_subj_corrupt:], batch_preds[num_subj_corrupt:], corrupted_objects], axis=1)

            neg_batch = np.concatenate([subject_based_corrupted_triples, object_based_corrupted_triples], axis=0)

            neg_batch = torch.tensor(neg_batch, dtype=torch.long, device=device)

            # Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            # model.zero_grad()
            # When to use model.zero_grad() and when optimizer.zero_grad() ?
            optimizer.zero_grad()

            loss = kg_embedding_model(pos_batch, neg_batch)

            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        stop = timeit.default_timer()
        log.info("Epoch %s took %s seconds \n" % (str(epoch), str(round(stop - start))))

    return kg_embedding_model
