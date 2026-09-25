Extending the Datasets
======================

While the core of PyKEEN uses the :class:`~pykeen.triples.TriplesFactory` for handling sets of triples, the definition
of a training, validation, and testing trichotomy for a given dataset is very useful for reproducible benchmarking. The
internal :class:`~pykeen.datasets.base.Dataset` class can be considered as a three-tuple of datasets (though it's
implemented as a class such that it can be extended). There are several datasets included in PyKEEN already, each coming
from sources that look different. This tutorial gives some insight into implementing your own Dataset class.

Pre-split Datasets
------------------

Unpacked Remote Dataset
~~~~~~~~~~~~~~~~~~~~~~~

Use this tutorial if you have three separate URLs for the respective training, testing, and validation sets that are
each 3 column TSV files. A good example can be found at https://github.com/ZhenfengLei/KGDatasets/tree/master/DBpedia50.
There's a base class called :class:`~pykeen.datasets.base.UnpackedRemoteDataset` that can be used to wrap it like the
following:

.. literalinclude:: /examples/howto/extending_datasets.py
    :lines: 4-21

Unsplit Datasets
----------------

Use this tutorial if you have a single URL for a TSV dataset that needs to be automatically split into training,
testing, and validation. A good example can be found at https://github.com/hetio/hetionet/raw/master/hetnet/tsv. There's
a base class called :class:`~pykeen.datasets.base.SingleTabbedDataset` that can be used to wrap it like the following:

.. literalinclude:: /examples/howto/extending_datasets.py
    :lines: 25-35

The value for `URL` can be anything that can be read by :func:`pandas.read_csv`. Additional options can be passed
through to the reading function, such as ``sep=','``, with the keyword argument ``read_csv_kwargs=dict(sep=',')``. Note
that the default separator for Pandas is a comma, but PyKEEN overrides it to be a tab, so you'll have to explicitly set
it if you want a comma. Since there's a random aspect to this process, you can also set the seed used for splitting with
the ``random_state`` keyword argument.

Combining a Source and a Loader
-------------------------------

The classes above are thin wrappers around two independent pieces, which you can also combine yourself when none of them
fits:

- A :class:`~pykeen.datasets.sources.Source` says *where the files are*. It resolves a set of logical file keys, e.g.,
  ``"training"``, to local paths, downloading and unpacking as needed: :class:`~pykeen.datasets.sources.LocalSource`,
  :class:`~pykeen.datasets.sources.RemoteSource` for one URL per split, and
  :class:`~pykeen.datasets.sources.TarArchiveSource` / :class:`~pykeen.datasets.sources.ZipArchiveSource` for members of
  a single archive.
- A :class:`~pykeen.datasets.loaders.Loader` says *how the files become triples factories*.
  :class:`~pykeen.datasets.loaders.PreSplitLoader` reads one file per split and makes the evaluation splits share the
  training split's entity and relation index, while :class:`~pykeen.datasets.loaders.AutoSplitLoader` reads a single
  table and splits it.

For example, a dataset whose three splits sit next to each other inside a zip archive, using a comma as the separator,
can be written as:

.. code-block:: python

    import pathlib

    from pykeen.datasets.base import PathDataset
    from pykeen.datasets.sources import ZipArchiveSource


    class MyDataset(PathDataset):
        """A dataset packed into a zip archive."""

        def __init__(self, cache_root: str | None = None, **kwargs):
            """Initialize the dataset."""
            self.cache_root = self._help_cache(cache_root)
            super().__init__(
                source=ZipArchiveSource(
                    members={
                        "training": pathlib.PurePath("my-data", "train.csv"),
                        "testing": pathlib.PurePath("my-data", "test.csv"),
                        "validation": pathlib.PurePath("my-data", "valid.csv"),
                    },
                    cache_root=self.cache_root,
                    url="https://example.org/my-data.zip",
                ),
                load_triples_kwargs={"delimiter": ","},
                **kwargs,
            )

If neither loader fits, e.g., because the data does not come from files at all, subclass
:class:`~pykeen.datasets.base.LazyDataset` and override its ``_load_factories`` method to return the mapping of splits
to triples factories directly. It is called at most once, on first access.

Updating the ``setup.cfg``
--------------------------

Whether you're making a pull request against PyKEEN or implementing a dataset in your own package, you can use Python
entrypoints to register your dataset with PyKEEN. Below is an example of the entrypoints that register
:class:`~pykeen.datasets.Hetionet`, :class:`~pykeen.datasets.DRKG`, and others that appear in the PyKEEN `setup.cfg
<https://github.com/pykeen/pykeen/blob/master/setup.cfg>`_. Under the ``pykeen.datasets`` header, you can pick whatever
name you want for the dataset as the key (appearing on the left side of the equals, e.g. ``hetionet``) and the path to
the class (appearing on the right side of the equals, e.g., ``pykeen.datasets.hetionet:Hetionet``). The right side is
constructed by the path to the module, the colon ``:``, then the name of the class.

.. code-block:: ini

    # setup.cfg
    ...
    [options.entry_points]
    console_scripts =
        pykeen = pykeen.cli:main
    pykeen.datasets =
        hetionet         = pykeen.datasets.hetionet:Hetionet
        conceptnet       = pykeen.datasets.conceptnet:ConceptNet
        drkg             = pykeen.datasets.drkg:DRKG
        ...

If you're working on a development version of PyKEEN, you also need to run ``pykeen readme`` in the shell to update the
README.md file.
