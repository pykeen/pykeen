# -*- coding: utf-8 -*-

"""Dataset utilities."""

import base64
import hashlib
import logging
import pathlib
import re
from typing import Any, Collection, Iterable, Mapping, Optional, Pattern, Tuple, Type, Union

import click
from tqdm import tqdm

from .base import Dataset, EagerDataset, PathDataset
from ..constants import PYKEEN_DATASETS
from ..triples import CoreTriplesFactory

logger = logging.getLogger(__name__)

dataset_regex_option = click.option("--dataset-regex", help="Regex for filtering datasets by name")
max_triples_option = click.option("--max-triples", type=int)
min_triples_option = click.option("--min-triples", type=int)


def iter_dataset_classes(
    regex_name_filter: Union[None, str, Pattern] = None,
    *,
    max_triples: Optional[int] = None,
    min_triples: Optional[int] = None,
    use_tqdm: bool = True,
) -> Iterable[Tuple[str, Type[Dataset]]]:
    """Iterate over dataset classes with given constraints.

    :param regex_name_filter: An optional regular expression string or pre-compiled regular expression
    :param max_triples: An optional maximum number of triples for the dataset
    :param min_triples: An optional minimum number of triples for the dataset
    :param use_tqdm: Should a progress bar be shown?
    :yields: Pairs of dataset names and dataset classes
    """
    from . import dataset_resolver

    it = sorted(
        dataset_resolver.lookup_dict.items(),
        key=Dataset.triples_pair_sort_key,
    )
    if max_triples is not None:
        it = [pair for pair in it if Dataset.triples_pair_sort_key(pair) <= max_triples]
    if min_triples is not None:
        it = [pair for pair in it if Dataset.triples_pair_sort_key(pair) >= min_triples]
    if regex_name_filter is not None:
        if isinstance(regex_name_filter, str):
            regex_name_filter = re.compile(regex_name_filter)
        it = [(name, dataset_cls) for name, dataset_cls in it if regex_name_filter.match(name)]
    it_tqdm = tqdm(
        it,
        desc="Datasets",
        disable=not use_tqdm,
    )
    for name, dataset_cls in it_tqdm:
        n_triples = Dataset.triples_sort_key(dataset_cls)
        it_tqdm.set_postfix(name=name, triples=f"{n_triples:,}")
        yield name, dataset_cls


def iter_dataset_instances(
    regex_name_filter: Union[None, str, Pattern] = None,
    *,
    max_triples: Optional[int] = None,
    min_triples: Optional[int] = None,
    use_tqdm: bool = True,
) -> Iterable[Tuple[str, Dataset]]:
    """Iterate over dataset instances with given constraints.

    :param regex_name_filter: An optional regular expression string or pre-compiled regular expression
    :param max_triples: An optional maximum number of triples for the dataset
    :param min_triples: An optional minimum number of triples for the dataset
    :param use_tqdm: Should a progress bar be shown?
    :yields: Pairs of dataset names and dataset instances
    """
    for name, cls in iter_dataset_classes(
        regex_name_filter=regex_name_filter,
        max_triples=max_triples,
        min_triples=min_triples,
        use_tqdm=use_tqdm,
    ):
        instance = get_dataset(dataset=cls)
        yield name, instance


def get_dataset(
    *,
    dataset: Union[None, str, pathlib.Path, Dataset, Type[Dataset]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    training: Union[None, str, pathlib.Path, CoreTriplesFactory] = None,
    testing: Union[None, str, pathlib.Path, CoreTriplesFactory] = None,
    validation: Union[None, str, pathlib.Path, CoreTriplesFactory] = None,
) -> Dataset:
    """Get a dataset, cached based on the given kwargs.

    :param dataset: The name of a dataset, an instance of a dataset, or the class for a dataset.
    :param dataset_kwargs: The keyword arguments, only to be used when a class for a dataset is used for
        the ``dataset`` keyword argument.
    :param training: A triples factory for training triples or a path to a training triples file if ``dataset=None``
    :param testing: A triples factory for testing triples or a path to a testing triples file  if ``dataset=None``
    :param validation: A triples factory for validation triples or a path to a validation triples file
        if ``dataset=None``
    :returns: An instantiated dataset

    :raises ValueError: for incorrect usage of the input of the function
    :raises TypeError: If a type is given for ``dataset`` but it's not a subclass of
        :class:`pykeen.datasets.Dataset`
    """
    from . import dataset_resolver, has_dataset

    if dataset is None and (training is None or testing is None):
        raise ValueError("Must specify either dataset or both training/testing triples factories")

    if dataset is not None and (training is not None or testing is not None):
        raise ValueError("Can not specify both dataset and training/testing triples factories.")

    if isinstance(dataset, Dataset):
        if dataset_kwargs:
            logger.warning("dataset_kwargs not used since a pre-instantiated dataset was given")
        return dataset

    if isinstance(dataset, pathlib.Path):
        return Dataset.from_path(dataset)

    # convert class to string to use caching
    if isinstance(dataset, type) and issubclass(dataset, Dataset):
        dataset = dataset_resolver.normalize_cls(cls=dataset)

    if isinstance(dataset, str):
        if has_dataset(dataset):
            return _cached_get_dataset(dataset, dataset_kwargs)
        else:
            # Assume it's a file path
            return Dataset.from_path(dataset)

    if dataset is not None:
        raise TypeError(f"Dataset is invalid type: {type(dataset)}")

    if isinstance(training, (str, pathlib.Path)) and isinstance(testing, (str, pathlib.Path)):
        if validation is None or isinstance(validation, (str, pathlib.Path)):
            return PathDataset(
                training_path=training,
                testing_path=testing,
                validation_path=validation,
                **(dataset_kwargs or {}),
            )
        elif validation is not None:
            raise TypeError(f"Validation is invalid type: {type(validation)}")

    if isinstance(training, CoreTriplesFactory) and isinstance(testing, CoreTriplesFactory):
        if validation is not None and not isinstance(validation, CoreTriplesFactory):
            raise TypeError(f"Validation is invalid type: {type(validation)}")
        if dataset_kwargs:
            logger.warning("dataset_kwargs are disregarded when passing pre-instantiated triples factories")
        return EagerDataset(
            training=training,
            testing=testing,
            validation=validation,
        )

    raise TypeError(
        f"""Training and testing must both be given as strings or Triples Factories.
        - Training: {type(training)}: {training}
        - Testing: {type(testing)}: {testing}
        """,
    )


def _digest_kwargs(dataset_kwargs: Mapping[str, Any], ignore: Collection[str] = tuple()) -> str:
    digester = hashlib.sha256()
    for key in sorted(dataset_kwargs.keys()):
        if key in ignore:
            continue
        digester.update(key.encode(encoding="utf8"))
        digester.update(str(dataset_kwargs[key]).encode(encoding="utf8"))
    return base64.urlsafe_b64encode(digester.digest()).decode("utf8")[:32]


def _set_inverse_triples_(dataset_instance: Dataset, create_inverse_triples: bool) -> Dataset:
    # note: we only need to set the create_inverse_triples in the training factory.
    dataset_instance.training.create_inverse_triples = create_inverse_triples
    if create_inverse_triples:
        dataset_instance.training.num_relations *= 2
    return dataset_instance


def _cached_get_dataset(
    dataset: str,
    dataset_kwargs: Optional[Mapping[str, Any]],
    force: bool = False,
) -> Dataset:
    """Get dataset by name, potentially using file-based cache."""
    from . import dataset_resolver

    # hash kwargs
    dataset_kwargs = dataset_kwargs or {}
    digest = _digest_kwargs(dataset_kwargs, ignore={"create_inverse_triples"})

    # normalize dataset name
    dataset_cls = dataset_resolver.lookup(dataset)
    dataset = dataset_resolver.normalize_cls(dataset_cls)

    # get canonic path
    path = PYKEEN_DATASETS.joinpath(dataset, "cache", digest)

    # try to use cached dataset
    if path.is_dir() and not force:
        logger.info(f"Loading cached preprocessed dataset from {path.as_uri()}")
        return _set_inverse_triples_(
            dataset_cls.from_directory_binary(path),
            create_inverse_triples=dataset_kwargs.get("create_inverse_triples", False),
        )

    # load dataset without cache
    dataset_instance = dataset_resolver.make(dataset, dataset_kwargs)

    # store cache
    logger.info(f"Caching preprocessed dataset to {path.as_uri()}")
    dataset_instance.to_directory_binary(path=path)

    return dataset_instance
