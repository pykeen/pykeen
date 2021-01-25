# -*- coding: utf-8 -*-

"""The Hetionet dataset.

Get a summary with ``python -m pykeen.datasets.hetionet``
"""

import logging

import click

from .base import SingleTabbedDataset
from ..typing import TorchRandomHint

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
        random_state: TorchRandomHint = 0,
        **kwargs,
    ):
        """Initialize the `Hetionet <https://github.com/hetio/hetionet>`_ dataset from [himmelstein2017]_.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param random_state: The random seed to use in splitting the dataset. Defaults to 0.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.SingleTabbedDataset`.
        """
        super().__init__(
            url=URL,
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
            **kwargs,
        )


@click.command()
def _main():
    ds = Hetionet()
    click.echo(ds.summary_str())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    _main()
