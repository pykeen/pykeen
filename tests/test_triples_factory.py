# -*- coding: utf-8 -*-

"""Test utils for processing numeric literals."""

import unittest

import numpy as np

from poem.instance_creation_factories.triples_numeric_literals_factory import TriplesNumericLiteralsFactory

triples = np.array(
    [
        ['peter', 'likes', 'chocolate_cake'],
        ['chocolate_cake', 'isA', 'dish'],
        ['susan', 'likes', 'pizza'],
    ],
    dtype=np.str,
)

numeric_triples = np.array(
    [
        ['peter', '/lit/hasAge', '30'],
        ['peter', '/lit/hasHeight', '185'],
        ['peter', '/lit/hasChildren', '2'],
        ['susan', '/lit/hasAge', '28'],
        ['susan', '/lit/hasHeight', '170'],
    ],
    dtype=np.str,
)


class NumericLiteralsUtilsTests(unittest.TestCase):
    """Class for testing utils for processing numeric literals.tsv."""

    def test_create_cwa_instances(self):
        """Test creating CWA instances."""
        factory = TriplesNumericLiteralsFactory(triples=triples, numeric_triples=numeric_triples)
        instances = factory.create_cwa_instances()

        literals = instances.multimodal_data['numeric_literals']
        literals_to_id = instances.data_relation_to_id

        id_peter = factory.entity_to_id['peter']
        id_age = literals_to_id['/lit/hasAge']
        id_height = literals_to_id['/lit/hasHeight']
        id_num_children = literals_to_id['/lit/hasChildren']

        self.assertEqual(literals[id_peter, id_age], 30)
        self.assertEqual(literals[id_peter, id_height], 185)
        self.assertEqual(literals[id_peter, id_num_children], 2)

        id_susan = factory.entity_to_id['susan']
        id_age = literals_to_id['/lit/hasAge']
        id_height = literals_to_id['/lit/hasHeight']
        id_num_children = literals_to_id['/lit/hasChildren']

        self.assertEqual(literals[id_susan, id_age], 28)
        self.assertEqual(literals[id_susan, id_height], 170)
        self.assertEqual(literals[id_susan, id_num_children], 0)

        id_chocolate_cake = factory.entity_to_id['chocolate_cake']
        id_age = literals_to_id['/lit/hasAge']
        id_height = literals_to_id['/lit/hasHeight']
        id_num_children = literals_to_id['/lit/hasChildren']

        self.assertEqual(literals[id_chocolate_cake, id_age], 0)
        self.assertEqual(literals[id_chocolate_cake, id_height], 0)
        self.assertEqual(literals[id_chocolate_cake, id_num_children], 0)
