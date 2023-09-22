# -*- coding: utf-8 -*-

"""Utility classes for constructing datasets."""

import logging
import pathlib
from typing import Any, Iterable, Mapping, Optional, Union

from pykeen.datasets.base import LazyDataset
from pykeen.triples import QuadruplesFactory

__all__ = ["TemporalPathDataset"]

logger = logging.getLogger(__name__)


class TemporalPathDataset(LazyDataset):
    """Temporal Lazy Dataset."""

    #: The actual instance of the training factory, which is exposed to the user through `training`
    _training: Optional[QuadruplesFactory] = None
    #: The actual instance of the testing factory, which is exposed to the user through `testing`
    _testing: Optional[QuadruplesFactory] = None
    #: The actual instance of the validation factory, which is exposed to the user through `validation`
    _validation: Optional[QuadruplesFactory] = None

    def __init__(
        self,
        training_path: Union[str, pathlib.Path],
        testing_path: Union[str, pathlib.Path],
        validation_path: Union[None, str, pathlib.Path],
        eager: bool = False,
        create_inverse_quadruples: bool = False,
        load_quadruples_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Init."""
        self.training_path = pathlib.Path(training_path)
        self.testing_path = pathlib.Path(testing_path)
        self.validation_path = pathlib.Path(validation_path) if validation_path else None

        self.create_inverse_quadruples = create_inverse_quadruples
        self.load_quadruples_kwargs = load_quadruples_kwargs

        if eager:
            self._load()
            self._load_validation()

    def _load(self) -> None:
        self._training = QuadruplesFactory.from_path(
            path=self.training_path,
            create_inverse_quadruples=self.create_inverse_quadruples,
            load_quadruples_kwargs=self.load_quadruples_kwargs,
        )

        self._testing = QuadruplesFactory.from_path(
            path=self.testing_path,
            entity_to_id=self._training.entity_to_id,
            relation_to_id=self._training.relation_to_id,
            timestamp_to_id=self._training.timestamp_to_id,
            create_inverse_quadruples=False,
            load_quadruples_kwargs=self.load_quadruples_kwargs,
        )

    def _load_validation(self) -> None:
        assert self._training is not None
        if self.validation_path is None:
            self._validation = None
        else:
            self._validation = QuadruplesFactory.from_path(
                path=self.validation_path,
                entity_to_id=self._training.entity_to_id,
                relation_to_id=self._training.relation_to_id,
                timestamp_to_id=self._training.timestamp_to_id,
                create_inverse_quadruples=False,
                load_quadruples_kwargs=self.load_quadruples_kwargs,
            )

    @property
    def training(self) -> QuadruplesFactory:  # type:ignore # noqa: D401
        """Return The training quadruples factory."""
        if not self._loaded:
            self._load()
        assert self._training is not None
        return self._training

    @property
    def testing(self) -> QuadruplesFactory:  # type:ignore # noqa: D401
        """The testing quadruples factory that shares indices with the training triples factory."""
        if not self._loaded:
            self._load()
        assert self._testing is not None
        return self._testing

    @property
    def validation(self) -> Optional[QuadruplesFactory]:  # type:ignore # noqa: D401
        """Return The validation quadruples factory that shares indices with the training triples factory."""
        if not self._loaded:
            self._load()
        if not self._loaded_validation:
            self._load_validation()
        return self._validation

    def _extra_repr(self) -> Iterable[str]:
        """Yield extra entries for the instance's string representation."""
        yield f"#entity={self.num_entities}"
        yield f"#relations={self.num_relations}"
        yield f"create_inverse={self.create_inverse_quadruples}"
