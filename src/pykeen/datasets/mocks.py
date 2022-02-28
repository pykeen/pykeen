"""Small mock datasets for testing."""
from .inductive.base import EagerInductiveDataset, InductiveDataset
from ..triples.generation import generate_triples_factory

__all__ = [
    "create_inductive_dataset",
]


def create_inductive_dataset(
    num_relations: int,
    num_entities_transductive: int,
    num_triples_training: int,
    num_entities_inductive: int,
    num_triples_inference: int,
    num_triples_testing: int,
    random_state: int = 42,
    create_inverse_triples: bool = False,
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
    :param random_state:
        the random state to use.

    :return:
        an inductive dataset with random triples
    """
    return EagerInductiveDataset(
        transductive_training=generate_triples_factory(
            num_entities=num_entities_transductive,
            num_relations=num_relations,
            num_triples=num_triples_training,
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
        ),
        inductive_inference=generate_triples_factory(
            num_entities=num_entities_inductive,
            num_relations=num_relations,
            num_triples=num_triples_inference,
            create_inverse_triples=create_inverse_triples,
            random_state=random_state + 1,  # different random states for different triples
        ),
        inductive_testing=generate_triples_factory(
            num_entities=num_entities_inductive,
            num_relations=num_relations,
            num_triples=num_triples_testing,
            create_inverse_triples=create_inverse_triples,
            random_state=random_state + 2,  # different random states for different triples
        ),
        create_inverse_triples=create_inverse_triples,
    )
