# -*- coding: utf-8 -*-

"""The inductive link prediction datasets from [teru2020]_.

- GitHub Repository: https://github.com/kkteru/grail
- Paper: https://arxiv.org/abs/1911.06962
"""

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import UnpackedRemoteDisjointInductiveDataset

__all__ = [
    "InductiveFB15k237",
    "InductiveWN18RR",
    "InductiveNELL",
]

BASE_URL = "https://raw.githubusercontent.com/kkteru/grail/master/data"

FB_TRAIN_URL = "{base_url}/fb237_{version}/train.txt"
FB_INDUCTIVE_INFERENCE_URL = "{base_url}/fb237_{version}_ind/train.txt"
FB_INDUCTIVE_VALIDATION_URL = "{base_url}/fb237_{version}_ind/valid.txt"
FB_INDUCTIVE_TEST_URL = "{base_url}/fb237_{version}_ind/test.txt"

WN_TRAIN_URL = "{base_url}/WN18RR_{version}/train.txt"
WN_INDUCTIVE_INFERENCE_URL = "{base_url}/WN18RR_{version}_ind/train.txt"
WN_INDUCTIVE_VALIDATION_URL = "{base_url}/WN18RR_{version}_ind/valid.txt"
WN_INDUCTIVE_TEST_URL = "{base_url}/WN18RR_{version}_ind/test.txt"

NELL_TRAIN_URL = "{base_url}/nell_{version}/train.txt"
NELL_INDUCTIVE_INFERENCE_URL = "{base_url}/nell_{version}_ind/train.txt"
NELL_INDUCTIVE_VALIDATION_URL = "{base_url}/nell_{version}_ind/valid.txt"
NELL_INDUCTIVE_TEST_URL = "{base_url}/nell_{version}_ind/test.txt"


# If GitHub ever gets upset from too many downloads, we can switch to
# the data posted at https://github.com/pykeen/pykeen/pull/154#issuecomment-730462039


@parse_docdata
class InductiveFB15k237(UnpackedRemoteDisjointInductiveDataset):
    """The inductive FB15k-237 dataset in 4 versions.

    ---
    name: FB15k-237
    citation:
        author: Teru
        year: 2020
        link: https://arxiv.org/abs/1911.06962
        github: kkteru/grail
    V1:
        transductive train entities: 1594
        relations: 180
        transductive train triples: 4245
        inductive inference entities: 1093
        inductive inference relations: 180
        inductive inference triples: 1993
        inductive validation triples: 206
        inductive test triples: 205
    V2:
        transductive train entities: 2608
        relations: 200
        transductive train triples: 9739
        inductive inference entities: 1660
        inductive inference relations: 200
        inductive inference triples: 4145
        inductive validation triples: 469
        inductive test triples: 478
    V3:
        transductive train entities: 3668
        relations: 215
        transductive train triples: 17986
        inductive inference entities: 2501
        inductive inference relations: 215
        inductive inference triples: 7406
        inductive validation triples: 866
        inductive test triples: 865
    V4:
        transductive train entities: 4707
        relations: 219
        transductive train triples: 27203
        inductive inference entities: 3051
        inductive inference relations: 219
        inductive inference triples: 11714
        inductive validation triples: 1416
        inductive test triples: 1424
    """

    def __init__(self, version: str = "v1", **kwargs):
        """Initialize a particular version of a dataset (out of 4) from [teru2020]_.

        :param version: v1 / v2 / v3 / v4 , differ in the sizes of train and inductive inference graphs
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            transductive_training_url=FB_TRAIN_URL.format(base_url=BASE_URL, version=version),
            inductive_inference_url=FB_INDUCTIVE_INFERENCE_URL.format(base_url=BASE_URL, version=version),
            inductive_validation_url=FB_INDUCTIVE_VALIDATION_URL.format(base_url=BASE_URL, version=version),
            inductive_testing_url=FB_INDUCTIVE_TEST_URL.format(base_url=BASE_URL, version=version),
            version=version,
            eager=True,
            **kwargs,
        )


@parse_docdata
class InductiveWN18RR(UnpackedRemoteDisjointInductiveDataset):
    """The inductive WN18RR dataset in 4 versions.

    ---
    name: WordNet-18 (RR)
    citation:
        author: Teru
        year: 2020
        link: https://arxiv.org/abs/1911.06962
        github: kkteru/grail
    V1:
        transductive train entities: 2746
        relations: 9
        transductive train triples: 5410
        inductive inference entities: 922
        inductive inference relations: 9
        inductive inference triples: 1618
        inductive validation triples: 185
        inductive test triples: 188
    V2:
        transductive train entities: 6954
        relations: 10
        transductive train triples: 15262
        inductive inference entities: 2757
        inductive inference relations: 10
        inductive inference triples: 4011
        inductive validation triples: 411
        inductive test triples: 411
    V3:
        transductive train entities: 12078
        relations: 11
        transductive train triples: 25901
        inductive inference entities: 5084
        inductive inference relations: 11
        inductive inference triples: 6327
        inductive validation triples: 538
        inductive test triples: 605
    V4:
        transductive train entities: 3861
        relations: 9
        transductive train triples: 7940
        inductive inference entities: 7084
        inductive inference relations: 9
        inductive inference triples: 12334
        inductive validation triples: 1394
        inductive test triples: 1429
    """

    def __init__(self, version: str = "v1", **kwargs):
        """Initialize a particular version of a dataset (out of 4) from [teru2020]_.

        :param version: v1 / v2 / v3 / v4 , differ in the sizes of train and inductive inference graphs
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            transductive_training_url=WN_TRAIN_URL.format(base_url=BASE_URL, version=version),
            inductive_inference_url=WN_INDUCTIVE_INFERENCE_URL.format(base_url=BASE_URL, version=version),
            inductive_validation_url=WN_INDUCTIVE_VALIDATION_URL.format(base_url=BASE_URL, version=version),
            inductive_testing_url=WN_INDUCTIVE_TEST_URL.format(base_url=BASE_URL, version=version),
            version=version,
            eager=True,
            **kwargs,
        )


@parse_docdata
class InductiveNELL(UnpackedRemoteDisjointInductiveDataset):
    """The inductive NELL dataset in 4 versions.

    ---
    name: NELL
    citation:
        author: Teru
        year: 2020
        link: https://arxiv.org/abs/1911.06962
        github: kkteru/grail
    V1:
        transductive train entities: 3103
        relations: 14
        transductive train triples: 4687
        inductive inference entities: 225
        inductive inference relations: 14
        inductive inference triples: 833
        inductive validation triples: 101
        inductive test triples: 100
    V2:
        transductive train entities: 2564
        relations: 88
        transductive train triples: 8219
        inductive inference entities: 2086
        inductive inference relations: 88
        inductive inference triples: 4586
        inductive validation triples: 459
        inductive test triples: 476
    V3:
        transductive train entities: 4647
        relations: 142
        transductive train triples: 16393
        inductive inference entities: 3566
        inductive inference relations: 142
        inductive inference triples: 8048
        inductive validation triples: 811
        inductive test triples: 809
    V4:
        transductive train entities: 2092
        relations: 76
        transductive train triples: 7546
        inductive inference entities: 2795
        inductive inference relations: 76
        inductive inference triples: 7073
        inductive validation triples: 716
        inductive test triples: 731
    """

    def __init__(self, version: str = "v1", **kwargs):
        """Initialize a particular version of a dataset (out of 4) from [teru2020]_.

        :param version: v1 / v2 / v3 / v4 , differ in the sizes of train and inductive inference graphs
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            transductive_training_url=NELL_TRAIN_URL.format(base_url=BASE_URL, version=version),
            inductive_inference_url=NELL_INDUCTIVE_INFERENCE_URL.format(base_url=BASE_URL, version=version),
            inductive_validation_url=NELL_INDUCTIVE_VALIDATION_URL.format(base_url=BASE_URL, version=version),
            inductive_testing_url=NELL_INDUCTIVE_TEST_URL.format(base_url=BASE_URL, version=version),
            version=version,
            eager=True,
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    for cls in [InductiveFB15k237, InductiveWN18RR, InductiveNELL]:
        click.secho(f"Loading {cls.__name__}", fg="green", bold=True)
        d = cls()
        d.summarize()


if __name__ == "__main__":
    _main()
