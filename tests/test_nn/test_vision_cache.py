"""Tests for the Wikidata image cache."""

import pathlib
import tempfile
from unittest import mock

from pykeen.nn.vision.cache import WikidataImageCache


def _entry(wikidata_id: str, relation: str, url: str) -> dict[str, dict[str, str]]:
    """Create a single SPARQL result entry."""
    return {
        "item": {"value": f"http://www.wikidata.org/entity/{wikidata_id}"},
        "relation": {"value": f"http://www.wikidata.org/prop/direct/{relation}"},
        "image": {"value": url},
    }


def test_get_image_paths_downloads_most_preferred_image_only() -> None:
    """Test that only one image per entity is downloaded, from the most preferred relation."""
    entries = [
        # flag image (P41) and image (P18) -> P18 is preferred
        _entry("Q1", "P41", "https://example.org/q1-flag.svg"),
        _entry("Q1", "P18", "https://example.org/q1-b.png"),
        _entry("Q1", "P18", "https://example.org/q1-a.jpg"),
        # only a logo image (P154)
        _entry("Q2", "P154", "https://example.org/q2-logo.png"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        module = mock.MagicMock()
        module.join.return_value = pathlib.Path(directory)
        with (
            mock.patch("pykeen.nn.text.cache.PYKEEN_MODULE.module", return_value=module),
            mock.patch.object(WikidataImageCache, "query", return_value=entries),
        ):
            cache = WikidataImageCache()
            cache.get_image_paths(ids=["Q1", "Q2", "Q3"])
    downloads = {(call.kwargs["name"], call.kwargs["url"]) for call in module.ensure.call_args_list}
    assert downloads == {
        ("Q1.jpg", "https://example.org/q1-a.jpg"),
        ("Q2.png", "https://example.org/q2-logo.png"),
    }
