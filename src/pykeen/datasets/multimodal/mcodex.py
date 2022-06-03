"""A simple multi-modal extension of CoDEx with entity descriptions from Wikidata."""
import itertools
import json
import pathlib
from typing import Any, Optional, Union

import click
import more_click
import pystow
from tqdm.auto import tqdm

from ..codex import CoDExSmall

_BASE_URL_PATTERN = "https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json?flavor=dump"


def safe_get(d: dict, *keys, default=None) -> Optional[Any]:
    for key in keys[:-1]:
        d = d.get(key, {})
    return d.get(keys[-1], default)


class MCodexSmall(CoDExSmall):
    """An extension of CoDExSmall, which adds textual entity descriptions."""

    def __init__(self, language: str = "en", force: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)

        # TODO: this should go to the dataset base class
        module = pystow.module("pykeen", "datasets", "mcodexsmall")

        # download textual data
        self.texts = {}
        for wikidata_id, i in tqdm(self.entity_to_id.items()):
            data = module.ensure_json("text", url=_BASE_URL_PATTERN.format(wikidata_id=wikidata_id), force=force)
            # compose textual description
            title = safe_get(data, "entities", wikidata_id, "labels", language, "value", default="unknown entity")
            description = safe_get(data, "entities", wikidata_id, "descriptions", language, "value")
            self.texts[i] = f"{title}: {description}"

    def to_directory_binary(self, path: Union[str, pathlib.Path]) -> None:
        super().to_directory_binary(path=path)
        with path.joinpath("text.json").open("w") as json_path:
            json.dump(self.texts, json_path)

    @classmethod
    def from_directory_binary(cls, path: Union[str, pathlib.Path]) -> "Dataset":
        eager = super().from_directory_binary(path=path)
        instance = cls.__new__(cls)
        instance._training = eager.training
        instance._validation = eager.validation
        instance._testing = eager.testing
        instance.metadata = eager.metadata
        with path.joinpath("text.json").open() as json_path:
            texts = json.load(json_path)
        instance.texts = texts
        return instance

    def summary_str(self, title: Optional[str] = None, show_examples: Optional[int] = 5, end="\n") -> str:
        s = super().summary_str(title=title, show_examples=show_examples, end=end)
        if not show_examples:
            return s
        return end.join(
            (
                s,
                *(f"{i}: {t}" for i, t in itertools.islice(self.texts.items(), show_examples)),
            )
        )


@click.command()
@more_click.verbose_option
def _main():
    from pykeen.datasets import get_dataset

    ds = get_dataset(dataset=MCodexSmall)
    ds.summarize(show_examples=10)


if __name__ == "__main__":
    _main()
