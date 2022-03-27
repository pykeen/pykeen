"""Datasets from the ILPC 2022 Challenge."""

from docdata import parse_docdata

from .base import UnpackedRemoteDisjointInductiveDataset

__all__ = [
    "ILPC2022Small",
    "ILPC2022Large",
]

# ZENODO_URL = "https://zenodo.org/record/6321299/files/pykeen/ilpc2022-v1.0.zip"

BASE_URL = "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data"
SMALL_TRAIN_URL = f"{BASE_URL}/small/train.txt"
SMALL_INFERENCE_URL = f"{BASE_URL}/small/inference.txt"
SMALL_INFERENCE_VAL_URL = f"{BASE_URL}/small/inference_validation.txt"
SMALL_INFERENCE_TEST_URL = f"{BASE_URL}/small/inference_test.txt"

LARGE_TRAIN_URL = f"{BASE_URL}/large/train.txt"
LARGE_INFERENCE_URL = f"{BASE_URL}/large/inference.txt"
LARGE_INFERENCE_VAL_URL = f"{BASE_URL}/large/inference_validation.txt"
LARGE_INFERENCE_TEST_URL = f"{BASE_URL}/large/inference_test.txt"


@parse_docdata
class ILPC2022Small(UnpackedRemoteDisjointInductiveDataset):
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
            transductive_training_url=SMALL_TRAIN_URL,
            inductive_inference_url=SMALL_INFERENCE_URL,
            inductive_validation_url=SMALL_INFERENCE_VAL_URL,
            inductive_testing_url=SMALL_INFERENCE_TEST_URL,
            create_inverse_triples=True,
            eager=True,
            **kwargs,
        )


@parse_docdata
class ILPC2022Large(UnpackedRemoteDisjointInductiveDataset):
    """An inductive link prediction dataset for the ILPC 2022 Challenge.

    ---
    name: ILPC2022 Large
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
            transductive_training_url=LARGE_TRAIN_URL,
            inductive_inference_url=LARGE_INFERENCE_URL,
            inductive_validation_url=LARGE_INFERENCE_VAL_URL,
            inductive_testing_url=LARGE_INFERENCE_TEST_URL,
            create_inverse_triples=True,
            eager=True,
            **kwargs,
        )


def _main():
    for cls in ILPC2022Small, ILPC2022Large:
        dataset = cls()
        dataset.summarize()


if __name__ == "__main__":
    _main()
