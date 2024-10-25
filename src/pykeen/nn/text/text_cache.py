"""Text cache utilities."""

from __future__ import annotations

import functools
import logging
import re
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from textwrap import dedent
from typing import Any, Callable, Literal, cast

import more_itertools
import requests
from class_resolver import ClassResolver

from ...constants import PYKEEN_MODULE
from ...utils import nested_get
from ...version import get_version

__all__ = [
    "text_cache_resolver",
    "TextCache",
    # Concrete classes
    "IdentityCache",
    "PyOBOTextCache",
    "WikidataTextCache",
]


logger = logging.getLogger(__name__)


class TextCache(ABC):
    """An interface for looking up text for various flavors of entity identifiers."""

    @abstractmethod
    def get_texts(self, identifiers: Sequence[str]) -> Sequence[str | None]:
        """Get text for the given identifiers for the cache."""


class IdentityCache(TextCache):
    """
    A cache without functionality.

    Mostly used for testing.
    """

    # docstr-coverage: inherited
    def get_texts(self, identifiers: Sequence[str]) -> Sequence[str | None]:  # noqa: D102
        return identifiers


PYOBO_PREFIXES_WARNED = set()


class PyOBOTextCache(TextCache):
    """A cache that looks up labels of biomedical entities based on their CURIEs."""

    def __init__(self, *args, **kwargs):
        """Instantiate the PyOBO cache, ensuring PyOBO is installed."""
        try:
            import pyobo
        except ImportError:
            raise ImportError(f"Can not use {self.__class__.__name__} because pyobo is not installed.") from None
        else:
            self._get_name = pyobo.get_name
        super().__init__(*args, **kwargs)

    def get_texts(self, identifiers: Sequence[str]) -> Sequence[str | None]:
        """Get text for the given CURIEs.

        :param identifiers:
            The compact URIs for each entity (e.g., ``['doid:1234', ...]``)

        :return:
            the label for each entity, looked up via :func:`pyobo.get_name`.
            Might be none if no label is available.
        """
        # This import doesn't need a wrapper since it's a transitive
        # requirement of PyOBO
        import bioregistry

        res: list[str | None] = []
        for curie in identifiers:
            try:
                prefix, identifier = curie.split(":", maxsplit=1)
            except ValueError:
                res.append(None)
                continue

            norm_prefix = bioregistry.normalize_prefix(prefix)
            if norm_prefix is None:
                if prefix not in PYOBO_PREFIXES_WARNED:
                    logger.warning("Prefix not registered in the Bioregistry: %s", prefix)
                    PYOBO_PREFIXES_WARNED.add(prefix)
                res.append(None)
                continue

            try:
                name = self._get_name(norm_prefix, identifier)
            except subprocess.CalledProcessError:
                if norm_prefix not in PYOBO_PREFIXES_WARNED:
                    logger.warning("could not get names from %s", norm_prefix)
                    PYOBO_PREFIXES_WARNED.add(norm_prefix)
                res.append(None)
                continue
            else:
                res.append(name)
        return res


class WikidataTextCache(TextCache):
    """A cache for requests against Wikidata's SPARQL endpoint."""

    #: Wikidata SPARQL endpoint. See https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service#Interfacing
    WIKIDATA_ENDPOINT = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"

    HEADERS: dict[str, str] = {
        # cf. https://meta.wikimedia.org/wiki/User-Agent_policy
        "User-Agent": (
            f"PyKEEN-Bot/{get_version()} (https://pykeen.github.io; pykeen2019@gmail.com) "
            f"requests/{requests.__version__}"
        ),
        # cf. https://wikitech.wikimedia.org/wiki/Robot_policy
        "Accept-Encoding": "gzip",
    }

    def __init__(self) -> None:
        """Initialize the cache."""
        self.module = PYKEEN_MODULE.module("wikidata")

    @staticmethod
    def verify_ids(ids: Sequence[str]):
        """
        Raise error if invalid IDs are encountered.

        :param ids:
            the ids to verify

        :raises ValueError:
            if any invalid ID is encountered
        """
        pattern = re.compile(r"Q(\d+)")
        invalid_ids = [one_id for one_id in ids if not pattern.match(one_id)]
        if invalid_ids:
            raise ValueError(f"Invalid IDs encountered: {invalid_ids}")

    @classmethod
    def query(
        cls,
        sparql: str | Callable[..., str],
        wikidata_ids: Sequence[str],
        batch_size: int = 256,
        timeout=None,
    ) -> Iterable[Mapping[str, Any]]:
        """
        Batched SPARQL query execution for the given IDS.

        :param sparql:
            the SPARQL query with a placeholder `ids`
        :param wikidata_ids:
            the Wikidata IDs
        :param batch_size:
            the batch size, i.e., maximum number of IDs per query
        :param timeout:
            the timeout for the GET request to the SPARQL endpoint

        :return:
            an iterable over JSON results, where the keys correspond to query variables,
            and the values to the corresponding binding
        """
        if not wikidata_ids:
            return {}

        if len(wikidata_ids) > batch_size:
            # break into smaller requests
            return chain.from_iterable(
                cls.query(sparql=sparql, wikidata_ids=id_batch, batch_size=batch_size)
                for id_batch in more_itertools.chunked(wikidata_ids, batch_size)
            )

        if isinstance(sparql, str):
            sparql = sparql.format
        sparql = sparql(ids=" ".join(f"wd:{i}" for i in wikidata_ids))
        logger.debug("running query: %s", sparql)
        if timeout is None:
            timeout = 60.0
        res = requests.get(
            cls.WIKIDATA_ENDPOINT,
            params={"query": sparql, "format": "json"},
            headers=cls.HEADERS,
            timeout=timeout,
        )
        res.raise_for_status()
        bindings = res.json()["results"]["bindings"]
        logger.debug(f"Retrieved {len(bindings)} bindings")
        return bindings

    @classmethod
    def query_text(
        cls, wikidata_ids: Sequence[str], language: str = "en", batch_size: int = 256
    ) -> Mapping[str, Mapping[str, str]]:
        """
        Query the SPARQL endpoints about information for the given IDs.

        :param wikidata_ids:
            the Wikidata IDs
        :param language:
            the label language
        :param batch_size:
            the batch size; if more ids are provided, break the big request into multiple smaller ones

        :return:
            a mapping from Wikidata Ids to dictionaries with the label and description of the entities
        """
        res_json = cls.query(
            sparql=functools.partial(
                dedent(
                    """
                        SELECT ?item ?itemLabel ?itemDescription WHERE {{{{
                            VALUES ?item {{ {ids} }}
                            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language}". }}
                        }}}}
                    """
                ).format,
                language=language,
            ),
            wikidata_ids=wikidata_ids,
            batch_size=batch_size,
        )
        result = {}
        for entry in res_json:
            wikidata_id = nested_get(entry, "item", "value", default="")
            assert isinstance(wikidata_id, str)  # for mypy
            wikidata_id = wikidata_id.rsplit("/", maxsplit=1)[-1]
            label = nested_get(entry, "itemLabel", "value", default="")
            assert isinstance(label, str)  # for mypy
            description = nested_get(entry, "itemDescription", "value", default="")
            assert isinstance(description, str)  # for mypy
            result[wikidata_id] = dict(label=label, description=description)
        return result

    def _load(self, wikidata_id: str, component: str) -> str | None:
        """Load information about a Wikidata ID from JSON file."""
        name = f"{wikidata_id}.json"
        if not self.module.join(name=name).is_file():
            return None
        return self.module.load_json(name=name)[component]

    def _save(self, entries: Mapping[str, Mapping[str, str]]):
        """Save entries as JSON."""
        logger.info(f"Saving {len(entries)} entries to {self.module}")
        for wikidata_id, entry in entries.items():
            name = f"{wikidata_id}.json"
            self.module.dump_json(name=name, obj=entry)

    def _get(self, ids: Sequence[str], component: Literal["label", "description"]) -> Sequence[str]:
        """
        Get the requested component for the given IDs.

        .. note ::
            this method uses file-based caching to avoid excessive requests to the Wikidata API.

        :param ids:
            the Wikidata IDs
        :param component:
            the selected component

        :return:
            the selected component for each Wikidata ID
        """
        self.verify_ids(ids=ids)
        # try to load cached first
        result: list[str | None] = [None] * len(ids)
        for i, wikidata_id in enumerate(ids):
            result[i] = self._load(wikidata_id=wikidata_id, component=component)
        # determine missing entries
        missing = [wikidata_id for wikidata_id, desc in zip(ids, result) if not desc]
        # retrieve information via SPARQL
        entries = self.query_text(wikidata_ids=missing)
        # save entries
        self._save(entries=entries)
        # fill missing descriptions
        w_to_i = {wikidata_id: i for i, wikidata_id in enumerate(ids)}
        for wikidata_id, entry in entries.items():
            result[w_to_i[wikidata_id]] = entry[component]
        # for mypy
        for item in result:
            assert isinstance(item, str)
        return cast(Sequence[str], result)

    def get_texts(self, identifiers: Sequence[str]) -> Sequence[str]:
        """Get a concatenation of the title and description for each Wikidata identifier.

        :param identifiers:
            the Wikidata identifiers, each starting with Q (e.g., ``['Q42']``)

        :return:
            the label and description for each Wikidata entity concatenated
        """
        # get labels & descriptions
        titles = self.get_labels(wikidata_identifiers=identifiers)
        descriptions = self.get_descriptions(wikidata_identifiers=identifiers)
        # compose labels
        return [f"{title}: {description}" for title, description in zip(titles, descriptions)]

    def get_labels(self, wikidata_identifiers: Sequence[str]) -> Sequence[str]:
        """
        Get entity labels for the given IDs.

        :param wikidata_identifiers:
            the Wikidata identifiers, each starting with Q (e.g., ``['Q42']``)

        :return:
            the label for each Wikidata entity
        """
        return self._get(ids=wikidata_identifiers, component="label")

    def get_descriptions(self, wikidata_identifiers: Sequence[str]) -> Sequence[str]:
        """
        Get entity descriptions for the given IDs.

        :param wikidata_identifiers:
            the Wikidata identifiers, each starting with Q (e.g., ``['Q42']``)

        :return:
            the description for each Wikidata entity
        """
        return self._get(ids=wikidata_identifiers, component="description")


#: A resolver for text caches
text_cache_resolver: ClassResolver[TextCache] = ClassResolver.from_subclasses(base=TextCache)
