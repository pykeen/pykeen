# -*- coding: utf-8 -*-

"""Implementation of basic utils for creating instances."""
from poem.constants import TRIPLES_FACTORY, NUMERIC_LITERALS_FACTORY, FACTORY_NAME
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.instance_creation_factories.triples_numeric_literals_factory import TriplesNumericLiteralsFactory

FACTORIES = {
    TRIPLES_FACTORY: TriplesFactory,
    NUMERIC_LITERALS_FACTORY: TriplesNumericLiteralsFactory
}


def get_factory(config) -> TriplesFactory:
    """."""

    factory_name = config[FACTORY_NAME]
    factory = FACTORIES.get(factory_name)

    if factory is None:
        raise ValueError(f'invalid factory name: {factory_name}')

    return factory(config)
