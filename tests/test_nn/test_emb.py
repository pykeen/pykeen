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

    def test_backwards_compatibility(self):
        """Test shape and num_embeddings."""
        assert self.instance.max_id == self.instance.num_embeddings
        assert self.instance.shape == (self.instance.embedding_dim,)
