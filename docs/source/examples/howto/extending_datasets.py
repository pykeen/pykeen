"""Extending the datasets."""

# %%
from pykeen.datasets.base import UnpackedRemoteDataset

TEST_URL = "https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/DBpedia50/test.txt"
TRAIN_URL = "https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/DBpedia50/train.txt"
VALID_URL = "https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/DBpedia50/valid.txt"


class DBpedia50(UnpackedRemoteDataset):
    """A subset of DBpedia."""

    def __init__(self, **kwargs):
        """Initialize the dataset."""
        super().__init__(
            training_url=TRAIN_URL,
            testing_url=TEST_URL,
            validation_url=VALID_URL,
            **kwargs,
        )


# %%
from pykeen.datasets.base import SingleTabbedDataset

URL = "https://github.com/hetio/hetionet/raw/master/hetnet/tsv/hetionet-v1.0-edges.sif.gz"


class Hetionet(SingleTabbedDataset):
    """The Hetionet biomedical knowledge graph."""

    def __init__(self, **kwargs):
        """Initialize the dataset."""
        super().__init__(url=URL, **kwargs)
