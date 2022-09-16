# -*- coding: utf-8 -*-

"""The Wikidata-based WD50K dataset (triple-only version) from [galkin2020]_.

- GitHub Repository: https://github.com/migalkin/StarE/tree/master/data/clean/wd50k/
- Paper: https://www.aclweb.org/anthology/2020.emnlp-main.596/

Get a summary with ``python -m pykeen.datasets.wd50k``,
"""

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import UnpackedRemoteDataset, HyperRelationalUnpackedRemoteDataset

BASE_URL = "https://raw.githubusercontent.com/migalkin/StarE/master/data/clean/wd50k/"
TRIPLES_VALID_URL = f"{BASE_URL}/triples/valid.txt"
TRIPLES_TEST_URL = f"{BASE_URL}/triples/test.txt"
TRIPLES_TRAIN_URL = f"{BASE_URL}/triples/train.txt"

HYPER_RELATIONAL_BASE_URL = "https://raw.githubusercontent.com/migalkin/StarE/master/data/clean/"
HYPER_RELATIONAL_MAIN_TRAIN_URL = f"{HYPER_RELATIONAL_BASE_URL}/wd50k/statements/train.txt"
HYPER_RELATIONAL_MAIN_VALID_URL = f"{HYPER_RELATIONAL_BASE_URL}/wd50k/statements/valid.txt"
HYPER_RELATIONAL_MAIN_TEST_URL = f"{HYPER_RELATIONAL_BASE_URL}/wd50k/statements/test.txt"

HYPER_RELATIONAL_33_TRAIN_URL = f"{HYPER_RELATIONAL_BASE_URL}/wd50k_33/statements/train.txt"
HYPER_RELATIONAL_33_VALID_URL = f"{HYPER_RELATIONAL_BASE_URL}/wd50k_33/statements/valid.txt"
HYPER_RELATIONAL_33_TEST_URL = f"{HYPER_RELATIONAL_BASE_URL}/wd50k_33/statements/test.txt"

HYPER_RELATIONAL_66_TRAIN_URL = f"{HYPER_RELATIONAL_BASE_URL}/wd50k_66/statements/train.txt"
HYPER_RELATIONAL_66_VALID_URL = f"{HYPER_RELATIONAL_BASE_URL}/wd50k_66/statements/valid.txt"
HYPER_RELATIONAL_66_TEST_URL = f"{HYPER_RELATIONAL_BASE_URL}/wd50k_66/statements/test.txt"

HYPER_RELATIONAL_100_TRAIN_URL = f"{HYPER_RELATIONAL_BASE_URL}/wd50k_100/statements/train.txt"
HYPER_RELATIONAL_100_VALID_URL = f"{HYPER_RELATIONAL_BASE_URL}/wd50k_100/statements/valid.txt"
HYPER_RELATIONAL_100_TEST_URL = f"{HYPER_RELATIONAL_BASE_URL}/wd50k_100/statements/test.txt"


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

    def __init__(self, **kwargs):
        """Initialize the WD50K (triples) dataset from [galkin2020]_.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            training_url=TRIPLES_TRAIN_URL,
            testing_url=TRIPLES_TEST_URL,
            validation_url=TRIPLES_VALID_URL,
            load_triples_kwargs={"delimiter": ","},
            **kwargs,
        )



@parse_docdata
class WD50K(HyperRelationalUnpackedRemoteDataset):
    """The hyper-relational version of WD50K.

    ---
    name: WD50K (hyper-relational)
    citation:
        author: Galkin
        year: 2020
        link: https://www.aclweb.org/anthology/2020.emnlp-main.596/
        arxiv: 2009.10847
        github: migalkin/StarE
    statistics:
        entities: 47,156 (5,460 qualifier-only)
        relations: 532 (45 qualifier-only)
        training: 166,435
        testing: 46,159
        validation: 23,913
        statements: 236,507
        statements w/ qualifiers: 32,167 (13.6%)
    """

    def __init__(self, **kwargs):
        """Initialize the WD50K (hyper-relational) dataset from [galkin2020]_.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            training_url=HYPER_RELATIONAL_MAIN_TRAIN_URL,
            testing_url=HYPER_RELATIONAL_MAIN_TEST_URL,
            validation_url=HYPER_RELATIONAL_MAIN_VALID_URL,
            load_triples_kwargs={"delimiter": ","},
            **kwargs,
        )

@parse_docdata
class WD50K_33(HyperRelationalUnpackedRemoteDataset):
    """The hyper-relational version of WD50K where 33% of statements have at lease one qualifier pair.

    ---
    name: WD50K (hyper-relational)
    citation:
        author: Galkin
        year: 2020
        link: https://www.aclweb.org/anthology/2020.emnlp-main.596/
        arxiv: 2009.10847
        github: migalkin/StarE
    statistics:
        entities: 38,124 (6,463 qualifier-only)
        relations: 475 (47 qualifier-only)
        training: 73,406
        testing: 18,133
        validation: 10,568
        statements: 102,107
        statements w/ qualifiers: 31,866 (31.2%)
    """

    def __init__(self, **kwargs):
        """Initialize the WD50K (33) (hyper-relational) dataset from [galkin2020]_.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            training_url=HYPER_RELATIONAL_33_TRAIN_URL,
            testing_url=HYPER_RELATIONAL_33_TEST_URL,
            validation_url=HYPER_RELATIONAL_33_VALID_URL,
            load_triples_kwargs={"delimiter": ","},
            **kwargs,
        )

@parse_docdata
class WD50K_66(HyperRelationalUnpackedRemoteDataset):
    """The hyper-relational version of WD50K where 66% of statements have at lease one qualifier pair.

    ---
    name: WD50K (hyper-relational)
    citation:
        author: Galkin
        year: 2020
        link: https://www.aclweb.org/anthology/2020.emnlp-main.596/
        arxiv: 2009.10847
        github: migalkin/StarE
    statistics:
        entities: 27,347 (7,167 qualifier-only)
        relations: 494 (53 qualifier-only)
        training: 35,968
        testing: 8,045
        validation: 5,154
        statements: 49,167
        statements w/ qualifiers: 31,696 (64.5%)
    """

    def __init__(self, **kwargs):
        """Initialize the WD50K (66) (hyper-relational) dataset from [galkin2020]_.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            training_url=HYPER_RELATIONAL_66_TRAIN_URL,
            testing_url=HYPER_RELATIONAL_66_TEST_URL,
            validation_url=HYPER_RELATIONAL_66_VALID_URL,
            load_triples_kwargs={"delimiter": ","},
            **kwargs,
        )

@parse_docdata
class WD50K_100(HyperRelationalUnpackedRemoteDataset):
    """The hyper-relational version of WD50K where 100% of statements have at lease one qualifier pair.

    ---
    name: WD50K (hyper-relational)
    citation:
        author: Galkin
        year: 2020
        link: https://www.aclweb.org/anthology/2020.emnlp-main.596/
        arxiv: 2009.10847
        github: migalkin/StarE
    statistics:
        entities: 18,792 (7,862 qualifier-only)
        relations: 279 (75 qualifier-only)
        training: 22,738
        testing: 5,297
        validation: 3,279
        statements: 31,314
        statements w/ qualifiers: 31,314 (100%)
    """

    def __init__(self, **kwargs):
        """Initialize the WD50K (100) (hyper-relational) dataset from [galkin2020]_.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            training_url=HYPER_RELATIONAL_100_TRAIN_URL,
            testing_url=HYPER_RELATIONAL_100_TEST_URL,
            validation_url=HYPER_RELATIONAL_100_VALID_URL,
            load_triples_kwargs={"delimiter": ","},
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    for cls in [WD50KT, WD50K, WD50K_33, WD50K_66, WD50K_100]:
        click.secho(f"Loading {cls.__name__}", fg="green", bold=True)
        d = cls()
        d.summarize()


if __name__ == "__main__":
    _main()
