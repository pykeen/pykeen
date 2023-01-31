# -*- coding: utf-8 -*-

"""New-style base module for all KGE models."""

from __future__ import annotations

import logging
from abc import ABC
from collections import defaultdict
from operator import itemgetter
from typing import (
    Any,
    ClassVar,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

import torch
from class_resolver import HintOrType, OptionalKwargs
from class_resolver.utils import OneOrManyHintOrType, OneOrManyOptionalKwargs, normalize_with_default
from torch import nn

from .base import Model
from ..nn import representation_resolver
from ..nn.modules import Interaction, interaction_resolver, parallel_unsqueeze
from ..nn.representation import Representation
from ..regularizers import Regularizer, regularizer_resolver
from ..triples import KGInfo
from ..typing import HeadRepresentation, InductiveMode, RelationRepresentation, TailRepresentation
from ..utils import check_shapes, get_batchnorm_modules

__all__ = [
    "_NewAbstractModel",
    "ERModel",
]

logger = logging.getLogger(__name__)


class _NewAbstractModel(Model, ABC):
    """An abstract class for knowledge graph embedding models (KGEMs).

    The only function that needs to be implemented for a given subclass is
    :meth:`Model.forward`. The job of the :meth:`Model.forward` function, as
    opposed to the completely general :meth:`torch.nn.Module.forward` is
    to take indices for the head, relation, and tails' respective representation(s)
    and to determine a score.

    Subclasses of Model can decide however they want on how to store entities' and
    relations' representations, how they want to be looked up, and how they should
    be scored. The :class:`ERModel` provides a commonly useful implementation
    which allows for the specification of one or more entity representations and
    one or more relation representations in the form of :class:`pykeen.nn.Embedding`
    as well as a matching instance of a :class:`pykeen.nn.Interaction`.
    """

    #: The default regularizer class
    regularizer_default: ClassVar[Optional[Type[Regularizer]]] = None
    #: The default parameters for the default regularizer class
    regularizer_default_kwargs: ClassVar[Optional[Mapping[str, Any]]] = None

    can_slice_h = True
    can_slice_r = True
    can_slice_t = True

    def _reset_parameters_(self):  # noqa: D401
        """Reset all parameters of the model in-place."""
        # cf. https://github.com/mberr/ea-sota-comparison/blob/6debd076f93a329753d819ff4d01567a23053720/src/kgm/utils/torch_utils.py#L317-L372   # noqa:E501
        # Make sure that all modules with parameters do have a reset_parameters method.
        uninitialized_parameters = set(map(id, self.parameters()))
        parents = defaultdict(list)

        # Recursively visit all sub-modules
        task_list = []
        for name, module in self.named_modules():
            # skip self
            if module is self:
                continue

            # Track parents for blaming
            for p in module.parameters():
                parents[id(p)].append(module)

            # call reset_parameters if possible
            if hasattr(module, "reset_parameters"):
                task_list.append((name.count("."), module))

        # initialize from bottom to top
        # This ensures that specialized initializations will take priority over the default ones of its components.
        for module in map(itemgetter(1), sorted(task_list, reverse=True, key=itemgetter(0))):
            module.reset_parameters()
            uninitialized_parameters.difference_update(map(id, module.parameters()))

        # emit warning if there where parameters which were not initialised by reset_parameters.
        if len(uninitialized_parameters) > 0:
            logger.warning(
                "reset_parameters() not found for all modules containing parameters. "
                "%d parameters where likely not initialized.",
                len(uninitialized_parameters),
            )

            # Additional debug information
            for i, p_id in enumerate(uninitialized_parameters, start=1):
                logger.debug("[%3d] Parents to blame: %s", i, parents.get(p_id))

    def _instantiate_regularizer(
        self,
        regularizer: HintOrType[Regularizer],
        regularizer_kwargs: OptionalKwargs = None,
    ) -> Optional[Regularizer]:
        """
        Instantiate a regularizer using the default if None is provided.

        The following precedence order is used:

        1. If the passed regularizer is not None, use it
        2. If the regularizer is None, use the default regularizer. In this case, the
           default kwargs will be used in favor of provided ones.
        3. If both, the regularizer and the default regularizer are None, return None.

        :param regularizer:
            the regularizer, or a hint thereof
        :param regularizer_kwargs:
            additional keyword-based parameters passed to the regularizer upon instantiation

        :return:
            the regularizer instance.
        """
        regularizer, regularizer_kwargs = normalize_with_default(
            choice=regularizer,
            kwargs=regularizer_kwargs,
            default=self.regularizer_default,
            default_kwargs=self.regularizer_default_kwargs,
        )
        return regularizer_resolver.make_safe(regularizer, regularizer_kwargs)

    def post_parameter_update(self) -> None:
        """Has to be called after each parameter update."""
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "post_parameter_update"):
                module.post_parameter_update()

    # docstr-coverage: inherited
    def collect_regularization_term(self):  # noqa: D102
        return sum(
            regularizer.pop_regularization_term()
            for regularizer in self.modules()
            if isinstance(regularizer, Regularizer)
        )


def _prepare_representation_module_list(
    max_id: int,
    shapes: Sequence[str],
    label: str,
    representations: OneOrManyHintOrType[Representation] = None,
    representations_kwargs: OneOrManyOptionalKwargs = None,
    skip_checks: bool = False,
) -> Sequence[Representation]:
    """
    Normalize list of representations and wrap into nn.ModuleList.

    .. note ::
        Important: use ModuleList to ensure that Pytorch correctly handles their devices and parameters

    :param representations:
        the representations, or hints for them.
    :param representations_kwargs:
        additional keyword-based parameters for instantiating representations from hints.
    :param max_id:
        the maximum representation ID. Newly instantiated representations will contain that many representations, and
        pre-instantiated ones have to provide at least that many.
    :param shapes:
        the symbolic shapes, which are used for shape verification, if skip_checks is False.
    :param label:
        a label to use for error messages (typically, "entities" or "relations").
    :param skip_checks:
        whether to skip shape verification.

    :return:
        a module list of instantiated representation modules.

    :raises ValueError:
        if the maximum ID or shapes do not match
    """
    # TODO: allow max_id being present in representation_kwargs; if it matches max_id
    # TODO: we could infer some shapes from the given interaction shape information
    rs = representation_resolver.make_many(representations, kwargs=representations_kwargs, max_id=max_id)

    # check max-id
    for r in rs:
        if r.max_id < max_id:
            raise ValueError(
                f"{r} only provides {r.max_id} {label} representations, but should provide {max_id}.",
            )
        elif r.max_id > max_id:
            logger.warning(
                f"{r} provides {r.max_id} {label} representations, although only {max_id} are needed."
                f"While this is not necessarily wrong, it can indicate an error where the number of {label} "
                f"representations was chosen wrong.",
            )

    rs = cast(Sequence[Representation], nn.ModuleList(rs))
    if skip_checks:
        return rs

    # check shapes
    if len(rs) != len(shapes):
        raise ValueError(
            f"Interaction function requires {len(shapes)} {label} representations, but {len(rs)} were given."
        )
    check_shapes(
        *zip(
            (r.shape for r in rs),
            shapes,
        ),
        raise_on_errors=True,
    )
    return rs


def repeat_if_necessary(
    scores: torch.FloatTensor,
    representations: Sequence[Representation],
    num: Optional[int],
) -> torch.FloatTensor:
    """
    Repeat score tensor if necessary.

    If a model does not have entity/relation representations, the scores for
    `score_{h,t}` / `score_r` are always the same. For efficiency, they are thus
    only computed once, but to meet the API, they have to be brought into the correct shape afterwards.

    :param scores: shape: (batch_size, ?)
        the score tensor
    :param representations:
        the representations. If empty (i.e. no representations for this 1:n scoring), repetition needs to be applied
    :param num:
        the number of times to repeat, if necessary.

    :return:
        the score tensor, which has been repeated, if necessary
    """
    if representations:
        return scores
    return scores.repeat(1, num)


class ERModel(
    Generic[HeadRepresentation, RelationRepresentation, TailRepresentation],
    _NewAbstractModel,
):
    """A commonly useful base for KGEMs using embeddings and interaction modules.

    This model does not use post-init hooks to automatically initialize all of its
    parameters. Rather, the call to :func:`Model.reset_parameters_` happens at the end of
    ``ERModel.__init__``. This is possible because all trainable parameters should necessarily
    be passed through the ``super().__init__()`` in subclasses of :class:`ERModel`.

    Other code can still be put after the call to ``super().__init__()`` in subclasses, such as
    registering regularizers (as done in :class:`pykeen.models.ConvKB` and :class:`pykeen.models.TransH`).
    ---
    citation:
        author: Ali
        year: 2021
        link: https://jmlr.org/papers/v22/20-825.html
        github: pykeen/pykeen
    """

    #: The entity representations
    entity_representations: Sequence[Representation]

    #: The relation representations
    relation_representations: Sequence[Representation]

    #: The weight regularizers
    weight_regularizers: List[Regularizer]

    #: The interaction function
    interaction: Interaction

    def __init__(
        self,
        *,
        triples_factory: KGInfo,
        interaction: HintOrType[Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation]],
        interaction_kwargs: OptionalKwargs = None,
        entity_representations: OneOrManyHintOrType[Representation] = None,
        entity_representations_kwargs: OneOrManyOptionalKwargs = None,
        relation_representations: OneOrManyHintOrType[Representation] = None,
        relation_representations_kwargs: OneOrManyOptionalKwargs = None,
        skip_checks: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the module.

        :param triples_factory:
            The triples factory facilitates access to the dataset.
        :param interaction: The interaction module (e.g., TransE)
        :param interaction_kwargs:
            Additional key-word based parameters given to the interaction module's constructor, if not already
            instantiated.
        :param entity_representations: The entity representation or sequence of representations
        :param entity_representations_kwargs:
            additional keyword-based parameters for instantiation of entity representations
        :param relation_representations: The relation representation or sequence of representations
        :param relation_representations_kwargs:
            additional keyword-based parameters for instantiation of relation representations
        :param skip_checks:
            whether to skip entity representation checks.
        :param kwargs:
            Keyword arguments to pass to the base model
        """
        # TODO: support "broadcasting" representation regularizers?
        # e.g. re-use the same regularizer for everything; or
        # pass a dictionary with keys "entity"/"relation";
        # values are either a regularizer hint (=the same regularizer for all repr); or a sequence of appropriate length
        super().__init__(triples_factory=triples_factory, **kwargs)
        self.interaction = interaction_resolver.make(interaction, pos_kwargs=interaction_kwargs)
        self.entity_representations = self._build_representations(
            triples_factory=triples_factory,
            representations=entity_representations,
            representations_kwargs=entity_representations_kwargs,
            label="entity",
            skip_checks=skip_checks,
        )
        self.relation_representations = self._build_representations(
            triples_factory=triples_factory,
            representations=relation_representations,
            representations_kwargs=relation_representations_kwargs,
            label="relation",
            skip_checks=skip_checks,
        )
        # Comment: it is important that the regularizers are stored in a module list, in order to appear in
        # model.modules(). Thereby, we can collect them automatically.
        self.weight_regularizers = nn.ModuleList()
        # Explicitly call reset_parameters to trigger initialization
        self.reset_parameters_()

    def _build_representations(
        self,
        triples_factory: KGInfo,
        representations: OneOrManyHintOrType[Representation] = None,
        representations_kwargs: OneOrManyOptionalKwargs = None,
        label: Literal["entity", "relation"] = "entity",
        **kwargs,
    ) -> Sequence[Representation]:
        """Build representations for the given factory."""
        # note, triples_factory is required instead of just using self.num_entities
        # and self.num_relations for the inductive case when this is different
        return _prepare_representation_module_list(
            representations=representations,
            representations_kwargs=representations_kwargs,
            max_id=triples_factory.num_entities if label == "entity" else triples_factory.num_relations,
            shapes=self.interaction.full_entity_shapes() if label == "entity" else self.interaction.relation_shape,
            label=label,
            **kwargs,
        )

    def append_weight_regularizer(
        self,
        parameter: Union[str, nn.Parameter, Iterable[Union[str, nn.Parameter]]],
        regularizer: HintOrType[Regularizer],
        regularizer_kwargs: OptionalKwargs = None,
        default_regularizer: HintOrType[Regularizer] = None,
        default_regularizer_kwargs: OptionalKwargs = None,
    ) -> None:
        """
        Add a model weight to a regularizer's weight list, and register the regularizer with the model.

        :param parameter:
            The parameter, either as name, or as nn.Parameter object. A list of available parameter names is shown by
             `sorted(dict(self.named_parameters()).keys())`.
        :param regularizer:
            the regularizer or a hint thereof
        :param regularizer_kwargs:
            additional keyword-based parameters for the regularizer's instantiation
        :param default_regularizer:
            the default regularizer; if None, use :attr:`regularizer_default`
        :param default_regularizer_kwargs:
            the default regularizer kwargs; if None, use :attr:`regularizer_default_kwargs`

        :raises KeyError: If an invalid parameter name was given
        """
        # instantiate regularizer
        regularizer = regularizer_resolver.make(
            *normalize_with_default(
                choice=regularizer,
                kwargs=regularizer_kwargs,
                default=default_regularizer or self.regularizer_default,
                default_kwargs=default_regularizer_kwargs or self.regularizer_default_kwargs,
            )
        )

        # normalize input
        if isinstance(parameter, (str, nn.Parameter)):
            parameter = [parameter]
        weights: Mapping[str, nn.Parameter] = dict(self.named_parameters())
        for param in parameter:
            if isinstance(param, str):
                if param not in weights:
                    raise KeyError(f"Invalid parameter_name={parameter}. Available are: {sorted(weights.keys())}.")
                param: nn.Parameter = weights[param]  # type: ignore
            regularizer.add_parameter(parameter=param)
        self.weight_regularizers.append(regularizer)

    def forward(
        self,
        h_indices: torch.LongTensor,
        r_indices: torch.LongTensor,
        t_indices: torch.LongTensor,
        slice_size: Optional[int] = None,
        slice_dim: int = 0,
        *,
        mode: Optional[InductiveMode],
    ) -> torch.FloatTensor:
        """Forward pass.

        This method takes head, relation and tail indices and calculates the corresponding scores.
        It supports broadcasting.

        :param h_indices:
            The head indices.
        :param r_indices:
            The relation indices.
        :param t_indices:
            The tail indices.
        :param slice_size:
            The slice size.
        :param slice_dim:
            The dimension along which to slice
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.

        :return:
            The scores

        :raises NotImplementedError:
            if score repetition becomes necessary
        """
        if not self.entity_representations or not self.relation_representations:
            raise NotImplementedError("repeat scores not implemented for general case.")
        h, r, t = self._get_representations(h=h_indices, r=r_indices, t=t_indices, mode=mode)
        return self.interaction.score(h=h, r=r, t=t, slice_size=slice_size, slice_dim=slice_dim)

    def score_hrt(self, hrt_batch: torch.LongTensor, *, mode: Optional[InductiveMode] = None) -> torch.FloatTensor:
        """Forward pass.

        This method takes head, relation and tail of each triple and calculates the corresponding score.

        :param hrt_batch: shape: (batch_size, 3), dtype: long
            The indices of (head, relation, tail) triples.
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.

        :return: shape: (batch_size, 1), dtype: float
            The score for each triple.
        """
        # Note: slicing cannot be used here: the indices for score_hrt only have a batch
        # dimension, and slicing along this dimension is already considered by sub-batching.
        # Note: we do not delegate to the general method for performance reasons
        # Note: repetition is not necessary here
        h, r, t = self._get_representations(h=hrt_batch[:, 0], r=hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode)
        return self.interaction.score_hrt(h=h, r=r, t=t)

    def _check_slicing(self, slice_size: Optional[int]) -> None:
        """Raise an error, if slicing is requested, but the model does not support it."""
        if not slice_size:
            return
        # batch normalization modules use batch statistics in training mode
        # -> different batch divisions lead to different results
        if self.training and get_batchnorm_modules(self):
            raise ValueError("This model does not support slicing, since it has batch normalization layers.")

    # docstr-coverage: inherited
    def score_t(
        self,
        hr_batch: torch.LongTensor,
        *,
        slice_size: Optional[int] = None,
        mode: Optional[InductiveMode] = None,
        tails: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        self._check_slicing(slice_size=slice_size)
        # add broadcast dimension
        hr_batch = hr_batch.unsqueeze(dim=1)
        h, r, t = self._get_representations(h=hr_batch[..., 0], r=hr_batch[..., 1], t=tails, mode=mode)
        # unsqueeze if necessary
        if tails is None or tails.ndimension() == 1:
            t = parallel_unsqueeze(t, dim=0)
        return repeat_if_necessary(
            scores=self.interaction.score(h=h, r=r, t=t, slice_size=slice_size, slice_dim=1),
            representations=self.entity_representations,
            num=self._get_entity_len(mode=mode) if tails is None else tails.shape[-1],
        )

    # docstr-coverage: inherited
    def score_h(
        self,
        rt_batch: torch.LongTensor,
        *,
        slice_size: Optional[int] = None,
        mode: Optional[InductiveMode] = None,
        heads: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        self._check_slicing(slice_size=slice_size)
        # add broadcast dimension
        rt_batch = rt_batch.unsqueeze(dim=1)
        h, r, t = self._get_representations(h=heads, r=rt_batch[..., 0], t=rt_batch[..., 1], mode=mode)
        # unsqueeze if necessary
        if heads is None or heads.ndimension() == 1:
            h = parallel_unsqueeze(h, dim=0)
        return repeat_if_necessary(
            scores=self.interaction.score(h=h, r=r, t=t, slice_size=slice_size, slice_dim=1),
            representations=self.entity_representations,
            num=self._get_entity_len(mode=mode) if heads is None else heads.shape[-1],
        )

    # docstr-coverage: inherited
    def score_r(
        self,
        ht_batch: torch.LongTensor,
        *,
        slice_size: Optional[int] = None,
        mode: Optional[InductiveMode] = None,
        relations: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        self._check_slicing(slice_size=slice_size)
        # add broadcast dimension
        ht_batch = ht_batch.unsqueeze(dim=1)
        h, r, t = self._get_representations(h=ht_batch[..., 0], r=relations, t=ht_batch[..., 1], mode=mode)
        # unsqueeze if necessary
        if relations is None or relations.ndimension() == 1:
            r = parallel_unsqueeze(r, dim=0)
        return repeat_if_necessary(
            scores=self.interaction.score(h=h, r=r, t=t, slice_size=slice_size, slice_dim=1),
            representations=self.relation_representations,
            num=self.num_relations if relations is None else relations.shape[-1],
        )

    def _get_entity_representations_from_inductive_mode(
        self, *, mode: Optional[InductiveMode]
    ) -> Sequence[Representation]:
        """
        Return the entity representations for the given inductive mode.

        :param mode:
            the inductive mode

        :raises ValueError:
            if the model does not support the given inductive mode, e.g.,
            because it is purely transductive

        :return:
            the entity representations for the given inductive mode
        """
        if mode is not None:
            raise ValueError(f"{self.__class__.__name__} does not support inductive mode: {mode}")
        return self.entity_representations

    def _get_entity_len(self, *, mode: Optional[InductiveMode]) -> Optional[int]:  # noqa:D105
        """
        Return the number of entities for the given inductive mode.

        :param mode:
            the inductive mode

        :raises NotImplementedError:
            if the model does not support the given inductive mode, e.g.,
            because it is purely transductive

        :return:
            the number of entities in the given inductive mode
        """
        if mode is not None:
            raise NotImplementedError
        return self.num_entities

    def _get_representations(
        self,
        h: Optional[torch.LongTensor],
        r: Optional[torch.LongTensor],
        t: Optional[torch.LongTensor],
        *,
        mode: Optional[InductiveMode],
    ) -> Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation]:
        """Get representations for head, relation and tails."""
        head_representations = tail_representations = self._get_entity_representations_from_inductive_mode(mode=mode)
        head_representations = [head_representations[i] for i in self.interaction.head_indices()]
        tail_representations = [tail_representations[i] for i in self.interaction.tail_indices()]
        hr, rr, tr = [
            [representation(indices=indices) for representation in representations]
            for indices, representations in (
                (h, head_representations),
                (r, self.relation_representations),
                (t, tail_representations),
            )
        ]
        # normalization
        return cast(
            Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation],
            tuple(x[0] if len(x) == 1 else x for x in (hr, rr, tr)),
        )
