# -*- coding: utf-8 -*-

"""Utility class for constructing inductive datasets."""

from typing import Optional, Mapping

from pykeen.triples import CoreTriplesFactory, TriplesFactory


class InductiveDataset():
    #: A factory wrapping the transductive training triples
    transductive_training: CoreTriplesFactory
    #: A factory wrapping the transductive testing triples, that share indices with the training triples
    transductive_testing: Optional[CoreTriplesFactory]
    #: A factory wrapping the transductive validation triples, that share indices with the training triples
    transductive_validation: Optional[CoreTriplesFactory]
    #: All datasets should take care of inverse triple creation
    create_inverse_triples: bool

    #: A factory wrapping the inductive inference triples
    inductive_inference: CoreTriplesFactory
    #: A factory wrapping the inductive testing triples
    inductive_testing: CoreTriplesFactory
    #: #: A factory wrapping the inductive validation inference triples
    inductive_validation: Optional[CoreTriplesFactory]

    @property
    def factory_dict(self) -> Mapping[str, CoreTriplesFactory]:
        """Return a dictionary of the three factories."""
        rv = dict(
            transductive_training=self.transductive_training,
            inductive_inference=self.inductive_inference,
            inductive_testing=self.inductive_testing,
        )
        if self.inductive_validation:
            rv['inductive_validation'] = self.inductive_validation
        if self.transductive_testing:
            rv['transductive_testing'] = self.transductive_testing
        if self.transductive_validation:
            rv['transductive_validation'] = self.transductive_validation
        return rv

    @property
    def transductive_training(self) -> TriplesFactory:  # type:ignore # noqa: D401
        """The transductive training triples factory."""
        if not self._loaded:
            self._load()
        assert self.transductive_training is not None
        return self.transductive_training

    @property
    def transductive_testing(self) -> TriplesFactory:  # type:ignore # noqa: D401
        """The transductive validation triples factory."""
        if not self._loaded:
            self._load()
        if not self._loaded_transductive_testing:
            self._load_transductive_testing()
        return self.transductive_testing

    @property
    def transductive_validation(self) -> TriplesFactory:  # type:ignore # noqa: D401
        """The transductive training triples factory."""
        if not self._loaded:
            self._load()
        if not self._loaded_transductive_validation:
            self._load_transductive_validation()
        return self.transductive_validation

    @property
    def inductive_inference(self) -> TriplesFactory:  # type:ignore # noqa: D401
        """The inductive inference triples factory."""
        if not self._loaded:
            self._load()
        assert self.inductive_inference is not None
        return self.inductive_inferenceg

    @property
    def inductive_validation(self) -> TriplesFactory:  # type:ignore # noqa: D401
        """The inductive testing triples factory."""
        if not self._loaded:
            self._load()
        if not self._loaded_inductive_validation:
            self._load_inductive_validation()
        return self.inductive_validation

    @property
    def inductive_testing(self) -> TriplesFactory:  # type:ignore # noqa: D401
        """The inductive testing triples factory."""
        if not self._loaded:
            self._load()
        assert self.inductive_testing is not None
        return self.inductive_testing

    @property
    def _loaded(self) -> bool:
        is_transductive_training = self.transductive_training is not None
        is_inductive_inference = self.inductive_inference is not None
        is_inductive_testing = self.inductive_testing is not None
        return is_transductive_training and is_inductive_inference and is_inductive_testing

    @property
    def _loaded_transductive_testing(self) -> bool:
        return self.transductive_training is not None

    @property
    def _loaded_transductive_validation(self) -> bool:
        return self.transductive_validation is not None

    @property
    def _loaded_inductive_validation(self) -> bool:
        return self.inductive_validation is not None

    def _load(self) -> None:
        NotImplementedError

    def _load_transductive_validation(self) -> None:
        NotImplementedError

    def _load_inductive_validation(self) -> None:
        NotImplementedError

    def _load_transductive_testing(self) -> None:
        NotImplementedError
