# -*- coding: utf-8 -*-

"""Test embeddings."""

from collections import ChainMap
from typing import Any, ClassVar, MutableMapping, Tuple

import numpy
import torch
import unittest_templates

import pykeen.nn.message_passing
import pykeen.nn.node_piece
import pykeen.nn.pyg
import pykeen.nn.representation
import pykeen.nn.utils
import pykeen.nn.vision
from pykeen.datasets import get_dataset
from tests import cases, constants, mocks

from ..utils import needs_packages


class EmbeddingTests(cases.RepresentationTestCase):
    """Tests for embeddings."""

    cls = pykeen.nn.representation.Embedding
    kwargs = dict(
        num_embeddings=7,
        embedding_dim=13,
    )

    def test_backwards_compatibility(self):
        """Test shape and num_embeddings."""
        assert self.instance.max_id == self.instance_kwargs["num_embeddings"]
        embedding_dim = int(numpy.prod(self.instance.shape))
        assert self.instance.shape == (embedding_dim,)


class LowRankEmbeddingRepresentationTests(cases.RepresentationTestCase):
    """Tests for low-rank embedding representations."""

    cls = pykeen.nn.representation.LowRankRepresentation
    kwargs = dict(
        shape=(3, 7),
    )

    def test_approximate(self):
        """Test approximation of other representations."""
        approx = self.cls.approximate(other=pykeen.nn.representation.Embedding(**self.instance_kwargs))
        assert isinstance(approx, self.cls)


class TensorEmbeddingTests(cases.RepresentationTestCase):
    """Tests for Embedding with 2-dimensional shape."""

    cls = pykeen.nn.representation.Embedding
    kwargs = dict(
        shape=(3, 7),
    )


class RGCNRepresentationTests(cases.TriplesFactoryRepresentationTestCase):
    """Test RGCN representations."""

    cls = pykeen.nn.message_passing.RGCNRepresentation
    num_bases: ClassVar[int] = 2

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["entity_representations_kwargs"] = dict(shape=self.num_entities)
        return kwargs


class TestSingleCompGCNRepresentationTests(cases.TriplesFactoryRepresentationTestCase):
    """Test single CompGCN representations."""

    cls = pykeen.nn.representation.SingleCompGCNRepresentation
    dim: ClassVar[int] = 3
    create_inverse_triples = True

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["combined"] = pykeen.nn.representation.CombinedCompGCNRepresentations(
            triples_factory=kwargs.pop("triples_factory"),
            entity_representations_kwargs=dict(embedding_dim=self.dim),
            relation_representations_kwargs=dict(embedding_dim=self.dim),
            dims=self.dim,
        )
        # inferred from triples factory
        kwargs.pop("max_id")
        return kwargs


class NodePieceRelationTests(cases.NodePieceTestCase):
    """Tests for node piece representation."""

    kwargs = dict(
        token_representations_kwargs=dict(
            shape=(3,),
        )
    )


class NodePieceAnchorTests(cases.NodePieceTestCase):
    """Tests for node piece representation with anchor nodes."""

    kwargs = dict(
        token_representations_kwargs=dict(
            shape=(3,),
        ),
        tokenizers="anchor",
        tokenizers_kwargs=dict(
            selection="degree",
        ),
    )


class NodePieceMixedTests(cases.NodePieceTestCase):
    """Tests for node piece representation with mixed tokenizers."""

    kwargs = dict(
        token_representations_kwargs=(
            dict(
                shape=(3,),
            ),
            dict(
                shape=(3,),
            ),
        ),
        tokenizers=("relation", "anchor"),
        num_tokens=(2, 3),
        tokenizers_kwargs=(
            dict(),
            dict(
                selection="degree",
            ),
        ),
    )

    def test_token_representations(self):
        """Verify that the number of token representations is correct."""
        assert len(self.instance.base) == 2


class TokenizationTests(cases.RepresentationTestCase):
    """Tests for tokenization representation."""

    cls = pykeen.nn.node_piece.TokenizationRepresentation
    vocabulary_size: int = 5
    num_tokens: int = 3

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["assignment"] = torch.randint(self.vocabulary_size, size=(self.max_id, self.num_tokens))
        kwargs["token_representation_kwargs"] = dict(shape=(self.vocabulary_size,))
        # inferred from assignment
        kwargs.pop("max_id")
        return kwargs


class SubsetRepresentationTests(cases.RepresentationTestCase):
    """Tests for subset representations."""

    cls = pykeen.nn.representation.SubsetRepresentation
    kwargs = dict(
        max_id=7,
    )
    shape: Tuple[int, ...] = (13,)

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["base"] = pykeen.nn.representation.Embedding(
            num_embeddings=2 * kwargs["max_id"],
            shape=self.shape,
        )
        return kwargs


class TextRepresentationTests(cases.RepresentationTestCase):
    """Test the label based representations."""

    cls = pykeen.nn.representation.TextRepresentation
    kwargs = dict(encoder="character-embedding")
    key_labels: str = "labels"

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        # the representation module infers the max_id from the provided labels
        kwargs.pop("max_id")
        dataset = get_dataset(dataset="nations")
        kwargs[self.key_labels] = sorted(dataset.entity_to_id.keys())
        self.max_id = dataset.num_entities
        return kwargs

    def test_from_dataset(self):
        """Test creating text-based representations from a dataset."""
        dataset = get_dataset(dataset="nations")
        kwargs = {key: value for key, value in self.instance_kwargs.items() if key != self.key_labels}
        instance = self.cls.from_dataset(dataset=dataset, **kwargs)
        assert instance.max_id == dataset.num_entities


class CachedTextRepresentationTests(TextRepresentationTests):
    """Tests for cached text representations."""

    cls = pykeen.nn.representation.CachedTextRepresentation
    kwargs = dict(encoder="character-embedding", cache=pykeen.nn.utils.IdentityCache())
    key_labels: str = "identifiers"


class SimpleMessagePassingRepresentationTests(cases.MessagePassingRepresentationTests):
    """Test for Pytorch Geometric representations using uni-relational message passing layers."""

    cls = pykeen.nn.pyg.SimpleMessagePassingRepresentation
    embedding_dim: int = 3
    kwargs = dict(
        base_kwargs=dict(shape=(embedding_dim,)),
        layers=["gcn"] * 2,
        layers_kwargs=dict(in_channels=embedding_dim, out_channels=embedding_dim),
    )


class TypedMessagePassingRepresentationTests(cases.MessagePassingRepresentationTests):
    """Test for Pytorch Geometric representations using categorical message passing layers."""

    cls = pykeen.nn.pyg.TypedMessagePassingRepresentation
    embedding_dim: int = 3
    kwargs = dict(
        base_kwargs=dict(shape=(embedding_dim,)),
        layers=["rgcn"],
        layers_kwargs=dict(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            num_bases=2,
            num_relations=cases.TriplesFactoryRepresentationTestCase.num_relations,
        ),
    )


class FeaturizedMessagePassingRepresentationTests(cases.MessagePassingRepresentationTests):
    """Test for Pytorch Geometric representations using categorical message passing layers."""

    cls = pykeen.nn.pyg.FeaturizedMessagePassingRepresentation
    embedding_dim: int = 3
    kwargs = dict(
        base_kwargs=dict(shape=(embedding_dim,)),
        layers=["gat"],
        layers_kwargs=dict(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            edge_dim=embedding_dim,  # should match relation dim
        ),
        relation_representation_kwargs=dict(
            shape=embedding_dim,
        ),
    )


@constants.skip_if_windows
@needs_packages("torchvision")
class VisualRepresentationTestCase(cases.RepresentationTestCase):
    """Tests for VisualRepresentation."""

    cls = pykeen.nn.vision.VisualRepresentation
    kwargs = dict(
        encoder="resnet18",
        layer_name="avgpool",
        transforms=[],
        trainable=False,
    )
    max_id = 7

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs)
        kwargs["images"] = list(torch.rand(self.max_id, 3, 28, 28))
        return kwargs


@constants.skip_if_windows
@needs_packages("torchvision")
class WikidataVisualRepresentationTestCase(cases.RepresentationTestCase):
    """Tests for Wikidata visual representations."""

    cls = pykeen.nn.vision.WikidataVisualRepresentation
    kwargs = dict(
        encoder="resnet18",
        layer_name="avgpool",
        trainable=False,
        wikidata_ids=[
            "Q1",
            "Q42",
            # the following entity does not have an image -> will have to use backfill
            "Q676",
        ],
    )

    # docstr-coverage: inherited
    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs)
        kwargs.pop("max_id")
        self.max_id = len(kwargs["wikidata_ids"])
        return kwargs


class CombinedRepresentationTestCase(cases.RepresentationTestCase):
    """Test for combined representations."""

    cls = pykeen.nn.representation.CombinedRepresentation
    kwargs = dict(
        base_kwargs=[
            dict(shape=(3,)),
            dict(shape=(4,)),
        ]
    )


class WikidataTextRepresentationTests(cases.RepresentationTestCase):
    """Tests for Wikidata text representations."""

    cls = pykeen.nn.representation.WikidataTextRepresentation
    kwargs = dict(
        identifiers=["Q100", "Q1000"],
        encoder="character-embedding",
    )

    # docstr-coverage: inherited
    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs)
        # the representation module infers the max_id from the provided labels
        kwargs.pop("max_id")
        self.max_id = len(kwargs["identifiers"])
        return kwargs


@needs_packages("pyobo")
class BiomedicalCURIERepresentationTests(cases.RepresentationTestCase):
    """Tests for biomedical CURIE representations."""

    cls = pykeen.nn.representation.BiomedicalCURIERepresentation
    kwargs = dict(
        identifiers=[
            "hgnc:12929",  # PCGF2
            "hgnc:391",  # AKT1
        ],
        encoder="character-embedding",
    )

    # docstr-coverage: inherited
    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs)
        # the representation module infers the max_id from the provided labels
        kwargs.pop("max_id")
        self.max_id = len(kwargs["identifiers"])
        return kwargs


class PartitionRepresentationTests(cases.RepresentationTestCase):
    """Tests for partition representation."""

    cls = pykeen.nn.representation.PartitionRepresentation
    max_ids: ClassVar[Tuple[int, ...]] = (5, 7)
    shape: Tuple[int, ...] = (3,)

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs)

        # max_id is inferred from assignment
        kwargs.pop("max_id")
        self.max_id = sum(self.max_ids)

        # create random assignment
        assignment = []
        for i, max_id in enumerate(self.max_ids):
            assignment.append(torch.stack([torch.full(size=(max_id,), fill_value=i), torch.arange(max_id)], dim=-1))
        assignment = torch.cat(assignment)
        assignment = assignment[torch.randperm(assignment.shape[0])]

        # update kwargs
        kwargs.update(
            dict(
                assignment=assignment,
                bases=[None] * len(self.max_ids),
                bases_kwargs=[dict(max_id=max_id, shape=self.shape) for max_id in self.max_ids],
            )
        )
        return kwargs

    def test_coherence(self):
        """Test coherence with base representations."""
        xs = self.instance(indices=None)
        for x, (repr_id, local_index) in zip(xs, self.instance.assignment):
            x_base = self.instance.bases[repr_id](indices=local_index)
            assert (x_base == x).all()

    def test_input_verification(self):
        """Verify that the input is correctly verified."""
        # empty bases
        with self.assertRaises(ValueError):
            self.cls(assignment=..., bases=[], bases_kwargs=[])

        # inconsistent base shapes
        shapes = range(2, len(self.max_ids) + 2)
        bases_kwargs = [dict(max_id=max_id, shape=(dim,)) for max_id, dim in zip(self.max_ids, shapes)]
        with self.assertRaises(ValueError):
            self.cls(**ChainMap(dict(bases_kwargs=bases_kwargs), self.instance_kwargs))

        # invalid base id
        assignment = self.instance.assignment.clone()
        assignment[torch.randint(assignment.shape[0], size=tuple()), 0] = len(self.instance.bases)
        with self.assertRaises(ValueError):
            self.cls(**ChainMap(dict(assignment=assignment), self.instance_kwargs))

        # invalid local index
        assignment = self.instance.assignment.clone()
        assignment[torch.randint(assignment.shape[0], size=tuple()), 1] = max(self.max_ids)
        with self.assertRaises(ValueError):
            self.cls(**ChainMap(dict(assignment=assignment), self.instance_kwargs))


class BackfillRepresentationTests(cases.RepresentationTestCase):
    """Tests for backfill representation, based on the partition representation."""

    cls = pykeen.nn.representation.BackfillRepresentation
    kwargs = dict(
        base_kwargs=dict(shape=(3,)),
        base_ids=[i for i in range(cases.RepresentationTestCase.max_id) if i % 2],
    )


class TransformedRepresentationTest(cases.RepresentationTestCase):
    """Tests for transformed representations."""

    cls = pykeen.nn.representation.TransformedRepresentation
    kwargs = dict(
        base_kwargs=dict(shape=(5,)),
    )

    # docstr-coverage: inherited
    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs)
        kwargs["transformation"] = torch.nn.Linear(5, 7)
        kwargs["base_kwargs"]["max_id"] = kwargs.pop("max_id")
        return kwargs


class TensorTrainRepresentationTest(cases.RepresentationTestCase):
    """Tests for tensor train representations."""

    cls = pykeen.nn.representation.TensorTrainRepresentation


class RepresentationModuleMetaTestCase(unittest_templates.MetaTestCase[pykeen.nn.representation.Representation]):
    """Test that there are tests for all representation modules."""

    base_cls = pykeen.nn.representation.Representation
    base_test = cases.RepresentationTestCase
    skip_cls = {mocks.CustomRepresentation, pykeen.nn.pyg.MessagePassingRepresentation}
