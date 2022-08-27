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

from docdata import parse_docdata

from .base import TarFileRemoteDataset

__all__ = [
    "Wikidata5M",
]


TRANSDUCTIVE_URL = "https://zenodo.org/record/5546383/files/wikidata5m_transductive.tar.gz"
INDUCTIVE_URL = "https://zenodo.org/record/5546387/files/wikidata5m_inductive.tar.gz"


@parse_docdata
class Wikidata5M(TarFileRemoteDataset):
    """The Wikidata5M dataset from [wang2019]_.

    ---
    name: Wikidata5M
    statistics:
        entities: 4594149
        relations: 822
        training: 20614279
        testing: 4977
        validation: 4983
        triples: 20624239
    citation:
        author: Wang
        year: 2019
        arxiv: 1911.06136
        link: https://arxiv.org/abs/1911.06136
    """

    def __init__(self, **kwargs):
        """Initialize the Wikidata5M dataset.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileRemoteDataset`.
        """
        super().__init__(
            url=TRANSDUCTIVE_URL,
            relative_training_path=pathlib.PurePath("wikidata5m_transductive_train.txt"),
            relative_testing_path=pathlib.PurePath("wikidata5m_transductive_test.txt"),
            relative_validation_path=pathlib.PurePath("wikidata5m_transductive_valid.txt"),
            **kwargs,
        )


if __name__ == "__main__":
    Wikidata5M.cli()
