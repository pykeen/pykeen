"""A module of vision related components.

Generally requires :mod:`torchvision` to be installed.
"""

import functools
import pathlib
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import torch
import torch.nn
import torch.utils.data
from class_resolver import OptionalKwargs
from docdata import parse_docdata
from typing_extensions import Self

from .cache import WikidataImageCache
from ..representation import BackfillRepresentation, Representation
from ..utils import ShapeError
from ...datasets import Dataset
from ...triples import TriplesFactory
from ...typing import FloatTensor, LongTensor, OneOrSequence

try:
    from PIL import Image
    from torchvision import models
    from torchvision import transforms as vision_transforms
except ImportError:
    models = vision_transforms = Image = None

__all__ = [
    "VisionDataset",
    "VisualRepresentation",
    "WikidataVisualRepresentation",
    "ImageHint",
    "ImageHints",
]


def _ensure_vision(instance: object, module: Any | None):
    if module is None:
        raise ImportError(f"{instance.__class__.__name__} requires `torchvision` to be installed.")


#: A path to an image file or a tensor representation of the image
ImageHint: TypeAlias = str | pathlib.Path | torch.Tensor
#: A sequence of image hints
ImageHints: TypeAlias = Sequence[ImageHint]


class VisionDataset(torch.utils.data.Dataset):
    """A dataset of images with data augmentation.

    .. note::

        requires :mod:`torchvision` to be installed.
    """

    def __init__(
        self,
        images: ImageHints,
        transforms: Sequence | None = None,
        root: pathlib.Path | None = None,
    ) -> None:
        """Initialize the dataset.

        :param images: The images, either as (relative) path, or preprocessed tensors.
        :param transforms: A sequence of transformations to apply to the images, cf. :mod:`torchvision.transforms`.
            Defaults to random size crops.
        :param root: The root directory for images.
        """
        _ensure_vision(self, vision_transforms)
        super().__init__()
        if root is None:
            root = pathlib.Path.cwd()
        self.root = pathlib.Path(root)

        self.images = images
        if transforms is None:
            transforms = [vision_transforms.RandomResizedCrop(size=224), vision_transforms.ToTensor()]
        transforms = list(transforms)
        transforms.append(vision_transforms.ConvertImageDtype(torch.get_default_dtype()))
        self.transforms = vision_transforms.Compose(transforms=transforms)

    # docstr-coverage: inherited
    def __getitem__(self, item: int) -> torch.Tensor:  # noqa:D105
        _ensure_vision(self, Image)
        image = self.images[item]
        if isinstance(image, str | pathlib.Path):
            path = pathlib.Path(image)
            if not path.is_absolute():
                path = self.root.joinpath(path)
            image = Image.open(path)
        assert isinstance(image, torch.Tensor | Image.Image)
        return self.transforms(image)

    # docstr-coverage: inherited
    def __len__(self) -> int:  # noqa:D105
        return len(self.images)


@parse_docdata
class VisualRepresentation(Representation):
    """Visual representations using a :mod:`torchvision` model.

    ---
    name: Visual
    """

    def __init__(
        self,
        images: ImageHints,
        encoder: str | torch.nn.Module,
        layer_name: str,
        max_id: int | None = None,
        shape: OneOrSequence[int] | None = None,
        transforms: Sequence | None = None,
        encoder_kwargs: OptionalKwargs = None,
        batch_size: int = 32,
        trainable: bool = True,
        **kwargs,
    ):
        """Initialize the representations.

        :param images: The images, either as tensors, or paths to image files.
        :param encoder: The encoder to use. If given as a string, lookup in :mod:`torchvision.models`.
        :param layer_name: The model's layer name to use for extracting the features, cf.
            :func:`torchvision.models.feature_extraction.create_feature_extractor`
        :param max_id: The number of representations. If given, it must match the number of images.
        :param shape: The shape of an individual representation. If provided, it must match the encoder output dimension
        :param transforms: Transformations to apply to the images. Notice that stochastic transformations will result in
            stochastic representations, too.
        :param encoder_kwargs: Additional keyword-based parameters passed to encoder upon instantiation.
        :param batch_size: The batch size to use during encoding.
        :param trainable: Whether the encoder should be trainable.
        :param kwargs: Additional keyword-based parameters passed to :class:`~pykeen.nn.representation.Representation`.

        :raises ValueError: If `max_id` is provided and does not match the number of images.
        """
        _ensure_vision(self, models)
        self.images = VisionDataset(images=images, transforms=transforms)

        if isinstance(encoder, str):
            cls = getattr(models, encoder)
            encoder = cls(encoder_kwargs or {})

        pool = functools.partial(torch.mean, dim=(-1, -2))

        encoder = models.feature_extraction.create_feature_extractor(
            model=encoder, return_nodes={layer_name: "feature"}
        )

        # infer shape
        with torch.inference_mode():
            encoder.eval()
            shape_ = self._encode(images=self.images[0].unsqueeze(dim=0), encoder=encoder, pool=pool).shape[1:]
        shape = ShapeError.verify(shape=shape_, reference=shape)
        if max_id is None:
            max_id = len(images)
        elif len(images) != max_id:
            raise ValueError(
                f"Inconsistent max_id={max_id} and len(images)={len(images)}. In case there are not images for all "
                f"IDs, you may consider using BackfillRepresentation.",
            )
        super().__init__(max_id=max_id, shape=shape, **kwargs)

        self.encoder = encoder
        self.pool = pool
        self.batch_size = batch_size or self.max_id
        self.encoder.train(trainable)
        self.encoder.requires_grad_(trainable)
        self.trainable = trainable

    @staticmethod
    def _encode(
        images: FloatTensor, encoder: torch.nn.Module, pool: Callable[[FloatTensor], FloatTensor]
    ) -> FloatTensor:
        """Encode images with the given encoder and pooling methods.

        :param images: shape: ``(batch_size, num_channels, height, width)`` A batch of images.
        :param encoder: The encoder, returning a dictionary with key "features".
        :param pool: The pooling method to use.

        :returns: shape: (batch_size, dim) The encoded representations.
        """
        return pool(encoder(images)["feature"])

    # docstr-coverage: inherited
    def _plain_forward(self, indices: LongTensor | None = None) -> FloatTensor:  # noqa: D102
        dataset = self.images
        if indices is not None:
            dataset = torch.utils.data.Subset(dataset=dataset, indices=indices)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
        # TODO: automatic batch size optimization?
        with torch.inference_mode(mode=not self.trainable):
            return torch.cat(
                [self._encode(images=images, encoder=self.encoder, pool=self.pool) for images in data_loader], dim=-1
            )


@parse_docdata
class WikidataVisualRepresentation(BackfillRepresentation):
    """Visual representations obtained from Wikidata and encoded with a vision encoder.

    If no image could be found for a certain Wikidata ID, a plain (trainable) embedding will be used instead.

    Example usage

    .. literalinclude:: ../examples/nn/representation/visual_wikidata.py

    ---
    name: Wikidata Visual
    """

    def __init__(
        self, wikidata_ids: Sequence[str], max_id: int | None = None, image_kwargs: OptionalKwargs = None, **kwargs
    ):
        """Initialize the representation.

        :param wikidata_ids: The Wikidata IDs.
        :param max_id: The total number of IDs. If provided, must match the length of ``wikidata_ids``.
        :param image_kwargs: Keyword-based parameters passed to
            :meth:`pykeen.nn.vision.cache.WikidataImageCache.get_image_paths`.
        :param kwargs: Additional keyword-based parameters passed to
            :class:`pykeen.nn.vision.representation.VisualRepresentation`.

        :raises ValueError: If the max_id does not match the number of Wikidata IDs.
        """
        max_id = max_id or len(wikidata_ids)
        if len(wikidata_ids) != max_id:
            raise ValueError(f"Inconsistent max_id={max_id} vs. len(wikidata_ids)={len(wikidata_ids)}")
        images = WikidataImageCache().get_image_paths(wikidata_ids, **(image_kwargs or {}))
        base_ids = [i for i, path in enumerate(images) if path is not None]
        images = [path for path in images if path is not None]
        super().__init__(
            max_id=max_id, base_ids=base_ids, base=VisualRepresentation, base_kwargs=dict(images=images, **kwargs)
        )

    @classmethod
    def from_triples_factory(
        cls,
        triples_factory: TriplesFactory,
        for_entities: bool = True,
        **kwargs,
    ) -> Self:
        """Prepare a visual representations for Wikidata entities from a triples factory.

        :param triples_factory: The triples factory.
        :param for_entities: Whether to create the initializer for entities (or relations).
        :param kwargs: Additional keyword-based arguments passed to
            :class:`pykeen.nn.vision.representation.WikidataVisualRepresentation`.

        :returns: A visual representation from the triples factory.
        """
        return cls(
            wikidata_ids=(triples_factory.entity_labeling if for_entities else triples_factory.relation_labeling)
            .all_labels()
            .tolist(),
            **kwargs,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        for_entities: bool = True,
        **kwargs,
    ) -> Self:
        """Prepare representations from a dataset.

        :param dataset: The dataset; needs to have Wikidata IDs as entity names.
        :param for_entities: Whether to create the initializer for entities (or relations).
        :param kwargs: Additional keyword-based arguments passed to
            :class:`pykeen.nn.vision.representation.WikidataVisualRepresentation`.

        :returns: A visual representation from the training factory in the dataset.

        :raises TypeError: If the triples factory does not provide labels.
        """
        if not isinstance(dataset.training, TriplesFactory):
            raise TypeError(f"{cls.__name__} requires access to labels, but dataset.training does not provide such.")
        return cls.from_triples_factory(triples_factory=dataset.training, for_entities=for_entities, **kwargs)
