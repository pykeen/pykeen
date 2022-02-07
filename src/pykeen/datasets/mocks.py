"""Small mock datasets for testing."""


from typing import Optional

from .inductive.base import EagerInductiveDataset, InductiveDataset
from ..triples.generation import generate_triples_factory


def create_inductive(
    num_relations: int,
    num_entities_transductive: int,
    num_triples_training: int,
    num_entities_inductive: Optional[int] = None,
    create_inverse_triples: bool = False,
    num_triples_inference: Optional[int] = None,
    num_triples_testing: Optional[int] = None,
    # num_triples_validation: Optional[int],
) -> InductiveDataset:
    """
    Create a random inductive dataset.

    :param num_relations:
        the number of relations
    :param num_entities_transductive:
        the number of entities in the transductive part
    :param num_triples_training:
        the number of (transductive) training triples
    :param num_entities_inductive:
        the number of entities in the inductive part. defaults to `num_entities_transductive`
    :param num_triples_inference:
        the number of (inductive) inference triples. defaults to `num_triples_training`
    :param num_triples_testing:
        the number of (inductive) testing triples. defaults to `num_triples_training`
    :param create_inverse_triples:
        whether to create inverse triples

    :return:
        an inductive dataset with random triples
    """
    return EagerInductiveDataset(
        transductive_training=generate_triples_factory(
            num_entities=num_entities_transductive,
            num_relations=num_relations,
            num_triples=num_triples_training,
            create_inverse_triples=create_inverse_triples,
        ),
        inductive_inference=generate_triples_factory(
            num_entities=num_entities_inductive,
            num_relations=num_relations,
            num_triples=num_triples_inference,
            create_inverse_triples=create_inverse_triples,
        ),
        inductive_testing=generate_triples_factory(
            num_entities=num_entities_inductive,
            num_relations=num_relations,
            num_triples=num_triples_testing,
            create_inverse_triples=create_inverse_triples,
        ),
        create_inverse_triples=create_inverse_triples,
    )
