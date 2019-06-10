import torch


def comute_nearest_neighbours(x, neighbours, k=10):
    """

    :param x: tensor of shape 1 x D
    :param neighbours: tensor of shape N x D
    :param k:
    :return:
    """
    dist = torch.nn.PairwiseDistance(p=2.0)
    x = x.expand(neighbours.shape[0], neighbours.shape[1])

    distances = dist(x, neighbours).view(-1)
    sorted_indices = torch.argsort(distances)

    k_nearest = distances[0:k]

    return sorted_indices[0:k], k_nearest
