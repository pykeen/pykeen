# -*- coding: utf-8 -*-

"""The Hetionet dataset.

Get a summary with ``python -m pykeen.datasets.hetionet``
"""

import logging
from typing import Union

import click
import numpy as np

from .base import SingleTabbedDataset

__all__ = [
    'Hetionet',
]

URL = 'https://github.com/hetio/hetionet/raw/master/hetnet/tsv/hetionet-v1.0-edges.sif.gz'


class Hetionet(SingleTabbedDataset):
    """The Hetionet dataset is a large biological network.

    In its publication [himmelstein2017]_, it is demonstrated to be useful for link prediction in drug repositioning
    and made publicly available through its `GitHub repository <https://github.com/hetio/hetionet>`_ in several formats.
    The link prediction algorithm showcased does not rely on embeddings, which leaves room for interesting comparison.
    One such comparison was made during the master's thesis of Lingling Xu [xu2019]_.

    For reproducibility, the random_state argument is set by default to 0. For permutation studies, you can change
    this.

    .. [himmelstein2017] Himmelstein, D. S., *et al* (2017). `Systematic integration of biomedical knowledge
       prioritizes drugs for repurposing <https://doi.org/10.7554/eLife.26726>`_. ELife, 6.
    .. [xu2019] Xu, L (2019) `A Comparison of Learned and Engineered Features in Network-Based Drug Repositioning
       <https://github.com/lingling93/master_thesis_drugrelink>`_. Master's Thesis.
    """

    def __init__(
        self,
        create_inverse_triples: bool = False,
        eager: bool = False,
        random_state: Union[None, int, np.random.RandomState] = 0,
    ):
        super().__init__(
            url=URL,
            eager=eager,
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
        )


@click.command()
def _main():
    ds = Hetionet()
    click.echo(ds.summary_str())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    _main()
