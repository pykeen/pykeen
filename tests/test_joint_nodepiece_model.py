from pykeen.datasets import FB15k237, Nations
from pykeen.losses import NSSALoss
from pykeen.models.unimodal import NodePiece

import unittest
import torch

from pykeen.nn import NodePieceRepresentation, RepresentationModule


class TestJointNodePiece(unittest.TestCase):

    def test_anchor_tokenizer(self):
        """
        The shape of NodePiece representation should correspond to number of anchors + 1 for AnchorTokenizer
        and number of relations*2 + 1 for RelationTokenizer
        """
        num_anchors = 5
        num_tokens = [3, 2]

        dataset = Nations(create_inverse_triples=True)
        model = NodePiece(
            triples_factory=dataset.training,
            random_seed=42,
            num_tokens=num_tokens,
            tokenizers=['anchor', 'relation'],
            tokenizers_kwargs=[
                dict(
                    selection="DegreeAnchorSelection",
                    searcher="ScipySparseAnchorSearcher",
                    selection_kwargs=dict(num_anchors=num_anchors),
                ),
                dict()
            ]
        )

        assert isinstance(model.entity_representations[0], NodePieceRepresentation)
        assert isinstance(model.entity_representations[0].tokens, torch.nn.ModuleList)
        assert model.entity_representations[0].tokens[0].max_id == num_anchors + 1
        assert model.entity_representations[0].tokens[1].max_id == dataset.num_relations + 1
