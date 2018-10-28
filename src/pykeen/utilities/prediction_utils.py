
import torch
from itertools import product
import numpy as np

from pykeen.utilities.triples_creation_utils.instance_creation_utils import create_mapped_triples


def create_triples(entity_pairs, relation):
    num_pairs = entity_pairs.size
    relation_repeated = np.repeat(relation, repeats=num_pairs)
    subjects = entity_pairs[:,0:1]
    objects = entity_pairs[:,1:2]

    triples = np.concatenate([subjects,relation_repeated,objects])
    return triples


def predict(kg_model, candidates, entity_to_id, rel_to_id, device):
    """

    :param kg_model: Trained KG model
    :param candidates: numpy array with two columns: 1.) Entites 2.) Relations
    :return:
    """

    entities = candidates[:,0:1]
    relations = candidates[:,1:2]

    all_entity_pairs = np.array(list(product(entities, entities)))
    all_triples = create_triples(entity_pairs=all_entity_pairs,relation=relations[0])

    for relation in relations[1:]:
        triples = create_triples(entity_pairs=all_entity_pairs,relation=relation)
        np.append(all_triples,triples)

    mapped_triples = create_mapped_triples(triples, entity_to_id=entity_to_id, rel_to_id=rel_to_id)

    mapped_triples = torch.tensor(mapped_triples, dtype=torch.long, device=device)

    predicted_scores = kg_model.predict(mapped_triples)

    _, sorted_indices = torch.sort(torch.tensor(predicted_scores, dtype=torch.float),
               descending=False)
    sorted_indices = sorted_indices.cpu().numpy

    ranked_triples = all_triples[:,sorted_indices]

    return ranked_triples
