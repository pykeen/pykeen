"""The Wikidata-based WD50K dataset (triple-only version) from [galkin2020]_.

- GitHub Repository: https://github.com/migalkin/StarE/tree/master/data/clean/wd50k/
- Paper: https://www.aclweb.org/anthology/2020.emnlp-main.596/

Get a summary with ``python -m pykeen.datasets.wd50k``,
"""

from typing import Unpack

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import UnpackedRemoteDataset, UnpackedRemoteDataSetKwargs

BASE_URL = "https://raw.githubusercontent.com/migalkin/StarE/master/data/clean/wd50k/"
TRIPLES_VALID_URL = f"{BASE_URL}/triples/valid.txt"
TRIPLES_TEST_URL = f"{BASE_URL}/triples/test.txt"
TRIPLES_TRAIN_URL = f"{BASE_URL}/triples/train.txt"


@parse_docdata
class WD50KT(UnpackedRemoteDataset):
    """The triples-only version of WD50K.

    ---
    name: WD50K (triples)
    citation:
        author: Galkin
        year: 2020
        link: https://www.aclweb.org/anthology/2020.emnlp-main.596/
        arxiv: 2009.10847
        github: migalkin/StarE
    statistics:
        entities: 40107
        relations: 473
        training: 164631
        testing: 45284
        validation: 22429
        triples: 232344
    """

    def __init__(self, **kwargs: Unpack[UnpackedRemoteDataSetKwargs]) -> None:
        """Initialize the WD50K (triples) dataset from [galkin2020]_."""
        kwargs["load_triples_kwargs"] = {"delimiter": ","}
        super().__init__(
            training_url=TRIPLES_TRAIN_URL,
            testing_url=TRIPLES_TEST_URL,
            validation_url=TRIPLES_VALID_URL,
            **kwargs,
        )


@click.command()
@verbose_option
def _main() -> None:
    for cls in [WD50KT]:
        click.secho(f"Loading {cls.__name__}", fg="green", bold=True)
        d = cls()
        d.summarize()


if __name__ == "__main__":
    _main()
