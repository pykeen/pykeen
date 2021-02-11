"""Test embeddings."""
from pykeen.nn import Embedding
from tests.cases import RepresentationTestCase


class EmbeddingTests(RepresentationTestCase):
    """Tests for embeddings."""

    cls = Embedding
    kwargs = dict(
        num_embeddings=7,
        embedding_dim=13,
    )
