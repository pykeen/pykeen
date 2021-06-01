# -*- coding: utf-8 -*-

"""The **Co**mpletion **D**atasets **Ex**tracted from Wikidata and Wikipedia (CoDEx) datasets from [safavi2020]_.

- GitHub Repository: https://github.com/tsafavi/codex
- Paper: https://arxiv.org/abs/2009.07810
"""

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import UnpackedRemoteDataset

BASE_URL = 'https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/'
SMALL_VALID_URL = f'{BASE_URL}/codex-s/valid.txt'
SMALL_TEST_URL = f'{BASE_URL}/codex-s/test.txt'
SMALL_TRAIN_URL = f'{BASE_URL}/codex-s/train.txt'

MEDIUM_VALID_URL = f'{BASE_URL}/codex-m/valid.txt'
MEDIUM_TEST_URL = f'{BASE_URL}/codex-m/test.txt'
MEDIUM_TRAIN_URL = f'{BASE_URL}/codex-m/train.txt'

LARGE_VALID_URL = f'{BASE_URL}/codex-l/valid.txt'
LARGE_TEST_URL = f'{BASE_URL}/codex-l/test.txt'
LARGE_TRAIN_URL = f'{BASE_URL}/codex-l/train.txt'


# If GitHub ever gets upset from too many downloads, we can switch to
# the data posted at https://github.com/pykeen/pykeen/pull/154#issuecomment-730462039

@parse_docdata
class CoDExSmall(UnpackedRemoteDataset):
    """The CoDEx small dataset.

    ---
    name: CoDEx (small)
    citation:
        author: Safavi
        year: 2020
        link: https://arxiv.org/abs/2009.07810
        github: tsafavi/codex
    statistics:
        entities: 2034
        relations: 42
        training: 32888
        testing: 1828
        validation: 1827
        triples: 36543
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the `CoDEx <https://github.com/tsafavi/codex>`_ small dataset from [safavi2020]_.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            training_url=SMALL_TRAIN_URL,
            testing_url=SMALL_TEST_URL,
            validation_url=SMALL_VALID_URL,
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


@parse_docdata
class CoDExMedium(UnpackedRemoteDataset):
    """The CoDEx medium dataset.

    ---
    name: CoDEx (medium)
    citation:
        author: Safavi
        year: 2020
        link: https://arxiv.org/abs/2009.07810
        github: tsafavi/codex
    statistics:
        entities: 17050
        relations: 51
        training: 185584
        testing: 10311
        validation: 10310
        triples: 206205
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the `CoDEx <https://github.com/tsafavi/codex>`_ medium dataset from [safavi2020]_.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            training_url=MEDIUM_TRAIN_URL,
            testing_url=MEDIUM_TEST_URL,
            validation_url=MEDIUM_VALID_URL,
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


@parse_docdata
class CoDExLarge(UnpackedRemoteDataset):
    """The CoDEx large dataset.

    ---
    name: CoDEx (large)
    citation:
        author: Safavi
        year: 2020
        link: https://arxiv.org/abs/2009.07810
        github: tsafavi/codex
    statistics:
        entities: 77951
        relations: 69
        training: 551193
        testing: 30622
        validation: 30622
        triples: 612437
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the `CoDEx <https://github.com/tsafavi/codex>`_ large dataset from [safavi2020]_.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            training_url=LARGE_TRAIN_URL,
            testing_url=LARGE_TEST_URL,
            validation_url=LARGE_VALID_URL,
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    for cls in [CoDExSmall, CoDExMedium, CoDExLarge]:
        click.secho(f'Loading {cls.__name__}', fg='green', bold=True)
        d = cls()
        d.summarize()


if __name__ == '__main__':
    _main()
