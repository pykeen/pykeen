import logging
import timeit

import numpy as np
import torch
import torch.optim as optim

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def train(kg_embedding_model, learning_rate, num_epochs, batch_size, pos_triples, neg_triples, device, seed):
    np.random.seed(seed=seed)
    indices = np.arange(pos_triples.shape[0])
    np.random.shuffle(indices)
    pos_triples = pos_triples[indices]
    neg_triples = neg_triples[indices]
    kg_embedding_model = kg_embedding_model.to(device)


    optimizer = optim.SGD(kg_embedding_model.parameters(), lr=learning_rate)

    total_loss = 0

    num_instances = pos_triples.shape[0]
    # num_batches = num_instances // num_epochs

    log.info('****Run Model On %s****' % str(device).upper())

    for epoch in range(num_epochs):
        start = timeit.default_timer()
        for step in range(num_instances):
            pos_triple = torch.tensor(pos_triples[step], dtype=torch.long, device=device)
            neg_triple = torch.tensor(neg_triples[step], dtype=torch.long, device=device)

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
