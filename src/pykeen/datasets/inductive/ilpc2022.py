"""Datasets from the ILPC 2022 Challenge."""

from docdata import parse_docdata

from .base import DisjointInductivePathDataset

__all__ = [
    "ILPC2022Small",
    "ILPC2022Large",
]


@parse_docdata
class ILPC2022Small(DisjointInductivePathDataset):
    """An inductive link prediction dataset for the ILPC 2022 Challenge.

    ---
    name: ILPC2022 Small
    citation:
        author: Galkin
        year: 2022
        link: https://arxiv.org/abs/2203.01520
        github: https://github.com/pykeen/ilpc2022
    """

    def __init__(self, **kwargs):
        """Initialize the inductive link prediction dataset.

        :param kwargs: keyword arguments to forward to the base dataset class, cf. DisjointInductivePathDataset
        """
        super().__init__(
            transductive_training_path=DATA.joinpath(size, "train.txt"),
            inductive_inference_path=DATA.joinpath(size, "inference.txt"),
            inductive_validation_path=DATA.joinpath(size, "inference_validation.txt"),
            inductive_testing_path=DATA.joinpath(size, "inference_test.txt"),
            create_inverse_triples=True,
            eager=True,
            **kwargs,
        )


@parse_docdata
class ILPC2022Large(DisjointInductivePathDataset):
    """An inductive link prediction dataset for the ILPC 2022 Challenge.

    ---
    name: ILPC2022 Small
    citation:
        author: Galkin
        year: 2022
        link: https://arxiv.org/abs/2203.01520
        github: https://github.com/pykeen/ilpc2022
    """

    def __init__(self, **kwargs):
        """Initialize the inductive link prediction dataset.

        :param kwargs: keyword arguments to forward to the base dataset class, cf. DisjointInductivePathDataset
        """
        super().__init__(
            transductive_training_path=DATA.joinpath(size, "train.txt"),
            inductive_inference_path=DATA.joinpath(size, "inference.txt"),
            inductive_validation_path=DATA.joinpath(size, "inference_validation.txt"),
            inductive_testing_path=DATA.joinpath(size, "inference_test.txt"),
            create_inverse_triples=True,
            eager=True,
            **kwargs,
        )
