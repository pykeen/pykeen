"""Image utilities for wikidata."""

from __future__ import annotations

import functools
import logging
import pathlib
from collections.abc import Collection, Mapping, Sequence
from textwrap import dedent

from tqdm.auto import tqdm

from ..text.cache import WikidataTextCache
from ...utils import nested_get, rate_limited

__all__ = [
    "WikidataImageCache",
]

logger = logging.getLogger(__name__)

WIKIDATA_IMAGE_RELATIONS = [
    "P18",  # image
    "P948",  # page banner
    "P41",  # flag image
    "P94",  # coat of arms image
    "P154",  # logo image
    "P242",  # locator map image
]

# TODO extract out a shared base class if we ever get a second image source


class WikidataImageCache(WikidataTextCache):
    """A cache for requests against Wikidata's SPARQL endpoint."""

    def _discover_images(self, extensions: Collection[str]) -> Mapping[str, pathlib.Path]:
        image_dir = self.module.join("images")
        return {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix in {f".{e}" for e in extensions}
        }

    def get_image_paths(
        self,
        ids: Sequence[str],
        extensions: Collection[str] = ("jpeg", "jpg", "gif", "png", "svg", "tif"),
        progress: bool = False,
    ) -> Sequence[pathlib.Path | None]:
        """Get paths to images for the given IDs.

        :param ids:
            the Wikidata IDs.
        :param extensions:
            the allowed file extensions
        :param progress:
            whether to display a progress bar

        :return:
            the paths to images for the given IDs.
        """
        id_to_path = self._discover_images(extensions=extensions)
        missing = sorted(set(ids).difference(id_to_path.keys()))
        num_missing = len(missing)
        logger.info(
            f"Downloading images for {num_missing:,} entities. With the rate limit in place, "
            f"this will take at least {num_missing / 10:.2f} seconds.",
        )
        res_json = self.query(
            sparql=functools.partial(
                dedent(
                    """
                    SELECT ?item ?relation ?image
                    WHERE {{
                        VALUES ?item {{ {ids} }} .
                        ?item ?r ?image .
                        VALUES ?r {{ {relations} }}
                    }}
                """
                ).format,
                relations=" ".join(f"wdt:{r}" for r in WIKIDATA_IMAGE_RELATIONS),
            ),
            wikidata_ids=missing,
        )
        # we can have multiple images per entity -> collect image URLs per image
        images: dict[str, dict[str, list[str]]] = {}
        for entry in res_json:
            # entity ID
            wikidata_id = nested_get(entry, "item", "value", default="")
            assert isinstance(wikidata_id, str)  # for mypy
            wikidata_id = wikidata_id.rsplit("/", maxsplit=1)[-1]

            # relation ID
            relation_id = nested_get(entry, "relation", "value", default="")
            assert isinstance(relation_id, str)  # for mypy
            relation_id = relation_id.rsplit("/", maxsplit=1)[-1]

            # image URL
            image_url = nested_get(entry, "image", "value", default=None)
            assert image_url is not None
            images.setdefault(wikidata_id, dict()).setdefault(relation_id, []).append(image_url)

        # check whether images are still missing
        missing = sorted(set(missing).difference(images.keys()))
        if missing:
            logger.warning(f"Could not retrieve an image URL for {len(missing)} entities: {missing}")

        # select on image url per image in a reproducible way
        for wikidata_id, url_dict in tqdm(rate_limited(images.items(), min_avg_time=0.1), disable=not progress):
            # traverse relations in order of preference
            for relation in WIKIDATA_IMAGE_RELATIONS:
                if relation not in url_dict:
                    continue
                # now there is an image available -> select reproducible by URL sorting
                image_url = sorted(url_dict[relation])[0]
                ext = image_url.rsplit(".", maxsplit=1)[-1].lower()
                if ext not in extensions:
                    logger.warning(f"Unknown extension: {ext} for {image_url}")
                self.module.ensure(
                    "images",
                    url=image_url,
                    name=f"{wikidata_id}.{ext}",
                    download_kwargs=dict(backend="requests", headers=self.HEADERS),
                )
            else:
                # did not break -> no image
                logger.warning(f"No image for {wikidata_id}")

        id_to_path = self._discover_images(extensions=extensions)
        return [id_to_path.get(i) for i in ids]
