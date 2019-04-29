# -*- coding: utf-8 -*-

"""Implementation of basic utils for creating instances."""
from poem.constants import DISTMULT_LITERAL_NAME_OWA, \
    KG_EMBEDDING_MODEL_NAME
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.instance_creation_factories.triples_numeric_literals_factory import TriplesNumericLiteralsFactory

KGE_MODEL_NAME_TO_FACTORY = {
    DISTMULT_LITERAL_NAME_OWA: TriplesNumericLiteralsFactory
}


def get_factory(config) -> TriplesFactory:
    """."""

    kge_model_name = config[KG_EMBEDDING_MODEL_NAME]
    factory = KGE_MODEL_NAME_TO_FACTORY.get(kge_model_name)

    if factory is None:
        raise ValueError(f'invalid factory name: {kge_model_name}')

    return factory(config)
