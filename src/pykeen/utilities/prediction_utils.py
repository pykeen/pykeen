# -*- coding: utf-8 -*-

from itertools import product

import numpy as np
import torch

from pykeen.utilities.triples_creation_utils.instance_creation_utils import create_mapped_triples


def create_triples(entity_pairs, relation):
    subjects = entity_pairs[:, 0:1]
    objects = entity_pairs[:, 1:2]
    relation_repeated = np.reshape(np.repeat(relation, repeats=subjects.shape[0]), newshape=(-1, 1))

    triples = np.concatenate([subjects, relation_repeated, objects], axis=1)

    return triples


def make_predictions(kg_model, entities, relations, entity_to_id, rel_to_id, device):
    """

    :param kg_model: Trained KG model
    :param candidates: numpy array with two columns: 1.) Entites 2.) Relations
    :return:
    """

    all_entity_pairs = np.array(list(product(entities, entities)))

    if relations.size == 1:
        all_triples = create_triples(entity_pairs=all_entity_pairs, relation=relations)
    else:
        all_triples = create_triples(entity_pairs=all_entity_pairs, relation=relations[0])

        for relation in relations[1:]:
            triples = create_triples(entity_pairs=all_entity_pairs, relation=relation)
            all_triples = np.append(all_triples, triples, axis=0)

    mapped_triples, _, _ = create_mapped_triples(all_triples, entity_to_id=entity_to_id, rel_to_id=rel_to_id)

    mapped_triples = torch.tensor(mapped_triples, dtype=torch.long, device=device)

    predicted_scores = kg_model.predict(mapped_triples)

    _, sorted_indices = torch.sort(torch.tensor(predicted_scores, dtype=torch.float),
                                   descending=False)
    sorted_indices = sorted_indices.cpu().numpy()

    ranked_triples = all_triples[sorted_indices, :]
    ranked_scores = np.reshape(predicted_scores[sorted_indices], newshape=(-1, 1))
    ranked_triples = np.concatenate([ranked_triples, ranked_scores], axis=1)

    return ranked_triples
