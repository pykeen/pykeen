# -*- coding: utf-8 -*-

"""Test utils for processing numeric literals"""

import unittest
import numpy as np
from poem.instance_creation_factories.triples_numeric_literals_factory import TriplesNumericLiteralsFactory
from poem.preprocessing.triples_preprocessing_utils.basic_triple_utils import create_entity_and_relation_mappings


class NumericLiteralsUtilsTests(unittest.TestCase):
    """Class for testing utils for processing numeric literals.tsv."""

    triples = np.array([['peter', 'likes', 'chocolate_cake'],
                        ['chocolate_cake', 'isA', 'dish'],
                        ['susan', 'likes', 'pizza']], dtype=np.str)

    literals = np.array([['peter', '/lit/hasAge', '30'],
                         ['peter', '/lit/hasHeight', '185'],
                         ['peter', '/lit/hasChildren', '2'],
                         ['susan', '/lit/hasAge', '28'],
                         ['susan', '/lit/hasHeight', '170'],
                         ], dtype=np.str)

    # literals_file = '/Users/mali/PycharmProjects/POEM_develop/tests/resources/numerical_literals.txt'

    def test_create_cwa_instances(self):
        """."""

        entity_to_id, relation_to_id = create_entity_and_relation_mappings(triples=self.triples)

        factory = TriplesNumericLiteralsFactory(entity_to_id=entity_to_id,
                                                relation_to_id=relation_to_id,
                                                numeric_triples=self.literals)
        instances = factory.create_cwa_instances(triples=self.triples)

        literals = instances.multimodal_data['numeric_literlas']
        literals_to_id = instances.data_relation_to_id

        id_peter = entity_to_id['peter']
        id_age = literals_to_id['/lit/hasAge']
        id_height = literals_to_id['/lit/hasHeight']
        id_num_children = literals_to_id['/lit/hasChildren']

        self.assertEqual(literals[id_peter, id_age], 30)
        self.assertEqual(literals[id_peter, id_height], 185)
        self.assertEqual(literals[id_peter, id_num_children], 2)

        id_susan = entity_to_id['susan']
        id_age = literals_to_id['/lit/hasAge']
        id_height = literals_to_id['/lit/hasHeight']
        id_num_children = literals_to_id['/lit/hasChildren']

        self.assertEqual(literals[id_susan, id_age], 28)
        self.assertEqual(literals[id_susan, id_height], 170)
        self.assertEqual(literals[id_susan, id_num_children], 0)

        id_chocolate_cake = entity_to_id['chocolate_cake']
        id_age = literals_to_id['/lit/hasAge']
        id_height = literals_to_id['/lit/hasHeight']
        id_num_children = literals_to_id['/lit/hasChildren']

        self.assertEqual(literals[id_chocolate_cake, id_age], 0)
        self.assertEqual(literals[id_chocolate_cake, id_height], 0)
        self.assertEqual(literals[id_chocolate_cake, id_num_children], 0)

        # Test creation of labels

        print(instances.labels)