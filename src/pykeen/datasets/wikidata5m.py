# -*- coding: utf-8 -*-

"""The Wikidata5m dataset from [wang2019]_.

Wikidata5m is a million-scale knowledge graph dataset with aligned corpus.
This dataset integrates the Wikidata knowledge graph and Wikipedia pages.
Each entity in Wikidata5m is described by a corresponding Wikipedia page,
which enables the evaluation of link prediction over unseen entities.

- Website: https://deepgraphlearning.github.io/project/wikidata5m
- Paper: https://arxiv.org/pdf/1911.06136.pdf

Get a summary with ``python -m pykeen.datasets.wikidata5m``,
"""

import pathlib

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import TarFileRemoteDataset

__all__ = [
    'Wikidata5mTransductive',
    'Wikidata5mInductive',
]

TRANSDUCTIVE_URL = 'https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1'
INDUCTIVE_URL = 'https://www.dropbox.com/s/csed3cgal3m7rzo/wikidata5m_inductive.tar.gz?dl=1'


@parse_docdata
class Wikidata5mTransductive(TarFileRemoteDataset):
    """The Wikidata5M Transductive dataset.

    ---
    name: Wikidata5M (Transductive)
    statistics:
        entities:
        relations:
        training:
        testing:
        validation:
        triples:
    citation:
        author: Wang
        year: 2019
        arxiv: 1911.06136
        link: https://arxiv.org/abs/1911.06136
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the Wikidata5M Transductive dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileRemoteDataset`.
        """
        super().__init__(
            url=TRANSDUCTIVE_URL,
            relative_training_path=pathlib.PurePath('wikidata5m_transductive', 'wikidata5m_transductive_train.txt'),
            relative_testing_path=pathlib.PurePath('wikidata5m_transductive', 'wikidata5m_transductive_test.txt'),
            relative_validation_path=pathlib.PurePath('wikidata5m_transductive', 'wikidata5m_transductive_valid.txt'),
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


@parse_docdata
class Wikidata5mInductive(TarFileRemoteDataset):
    """The Wikidata5M Inductive dataset.

    ---
    name: Wikidata5M (Inductive)
    statistics:
        entities:
        relations:
        training:
        testing:
        validation:
        triples:
    citation:
        author: Wang
        year: 2019
        arxiv: 1911.06136
        link: https://arxiv.org/abs/1911.06136
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the Wikidata5M Inductive dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileRemoteDataset`.
        """
        super().__init__(
            url=INDUCTIVE_URL,
            relative_training_path=pathlib.PurePath('wikidata5m_inductive', 'wikidata5m_inductive_train.txt'),
            relative_testing_path=pathlib.PurePath('wikidata5m_inductive', 'wikidata5m_inductive_test.txt'),
            relative_validation_path=pathlib.PurePath('wikidata5m_inductive', 'wikidata5m_inductive_valid.txt'),
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    for cls in (Wikidata5mTransductive, Wikidata5mInductive):
        ds = cls()
        ds.summarize()


if __name__ == "__main__":
    _main()
