import logging
import timeit

import numpy as np
import torch
import torch.optim as optim

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def split_list_in_batches(input_list, batch_size):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]


def train(kg_embedding_model, learning_rate, num_epochs, batch_size, pos_triples, neg_triples, device, seed):
    np.random.seed(seed=seed)
    indices = np.arange(pos_triples.shape[0])
    np.random.shuffle(indices)
    pos_triples = pos_triples[indices]
    neg_triples = neg_triples[indices]
    kg_embedding_model = kg_embedding_model.to(device)

    optimizer = optim.SGD(kg_embedding_model.parameters(), lr=learning_rate)

    total_loss = 0

    log.info('****Run Model On %s****' % str(device).upper())

    for epoch in range(num_epochs):
        start = timeit.default_timer()
        pos_batches = split_list_in_batches(input_list=pos_triples, batch_size=batch_size)
        neg_batches = split_list_in_batches(input_list=neg_triples, batch_size=batch_size)
        for i in range(len(pos_batches)):
            pos_batch = pos_batches[i]
            neg_batch = neg_batches[i]
            pos_triple = torch.tensor(pos_batch, dtype=torch.long, device=device)
            neg_triple = torch.tensor(neg_batch, dtype=torch.long, device=device)

            # Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            # model.zero_grad()
            # When to use model.zero_grad() and when optimizer.zero_grad() ?
            optimizer.zero_grad()

            loss = kg_embedding_model(pos_triple, neg_triple)

            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        stop = timeit.default_timer()
        log.info("Epoch %s took %s seconds \n" % (str(epoch), str(round(stop - start))))

    return kg_embedding_model
