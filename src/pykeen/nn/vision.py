"""
A module of vision related components.

Generally requires :module:`torchvision` to be installed.
"""
import functools
import pathlib
from typing import Any, Callable, Optional, Sequence, Union

import torch
import torch.nn
import torch.utils.data
from class_resolver import OptionalKwargs

from .representation import BackfillRepresentation, Representation
from .utils import ShapeError, WikidataCache
from ..datasets import Dataset
from ..triples import TriplesFactory
from ..typing import OneOrSequence

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
]


def _ensure_vision(instance: object, module: Optional[Any]):
    if module is None:
        raise ImportError(f"{instance.__class__.__name__} requires `torchvision` to be installed.")


class VisionDataset(torch.utils.data.Dataset):
    """
    A dataset of images with data augmentation.

    .. note ::
        requires `torchvision` to be installed.
    """

    def __init__(
        self,
        images: Sequence[Union[str, pathlib.Path, torch.Tensor]],
        transforms: Optional[Sequence] = None,
        root: Optional[pathlib.Path] = None,
    ) -> None:
        """
        Initialize the dataset.

        :param images: the images, either as (relative) path, or preprocessed tensors.
        :param transforms:
            a sequence of transformations to apply to the images,
            cf. :module:`torchvision.transforms`
        :param root:
            the root directory for images
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
        if isinstance(image, (str, pathlib.Path)):
            path = pathlib.Path(image)
            if not path.is_absolute():
                path = self.root.joinpath(path)
            image = Image.open(path)
        assert isinstance(image, (torch.Tensor, Image.Image))
        return self.transforms(image)

    # docstr-coverage: inherited
    def __len__(self) -> int:  # noqa:D105
        return len(self.images)


class VisualRepresentation(Representation):
    """Visual representations using a torchvision model."""

    def __init__(
        self,
        images: Sequence,
        encoder: Union[str, torch.nn.Module],
        layer_name: str,
        max_id: Optional[int] = None,
        shape: Optional[OneOrSequence[int]] = None,
        transforms: Optional[Sequence] = None,
        encoder_kwargs: OptionalKwargs = None,
        batch_size: int = 32,
        trainable: bool = True,
        **kwargs,
    ):
        """
        Initialize the representations.

        :param images:
            the images, either as tensors, or paths to image files.
        :param encoder:
            the encoder to use. If given as a string, lookup in :module:`torchvision.models`
        :param layer_name:
            the model's layer name to use for extracting the features, cf.
            :func:`torchvision.models.feature_extraction.create_feature_extractor`
        :param max_id:
            the number of representations. If given, it must match the number of images.
        :param shape:
            the shape of an individual representation. If provided, it must match the encoder output dimension
        :param transforms:
            transformations to apply to the images. Notice that stochastic transformations will result in
            stochastic representations, too.
        :param encoder_kwargs:
            additional keyword-based parameters passed to encoder upon instantiation.
        :param batch_size:
            the batch size to use during encoding
        :param trainable:
            whether the encoder should be trainable
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Representation.__init__`.

        :raises ValueError:
            if `max_id` is provided and does not match the number of images
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
        images: torch.FloatTensor, encoder: torch.nn.Module, pool: Callable[[torch.FloatTensor], torch.FloatTensor]
    ) -> torch.FloatTensor:
        """
        Encode images with the given encoder and pooling methods.

        :param images: shape: (batch_size, num_channels, height, width)
            a batch of images
        :param encoder:
            the encoder, returning a dictionary with key "features"
        :param pool:
            the pooling method to use
        :return: shape: (batch_size, dim)
            the encoded representations.
        """
        return pool(encoder(images)["feature"])

    # docstr-coverage: inherited
    def _plain_forward(self, indices: Optional[torch.LongTensor] = None) -> torch.FloatTensor:  # noqa: D102
        dataset = self.images
        if indices is not None:
            dataset = torch.utils.data.Subset(dataset=dataset, indices=indices)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
        with torch.inference_mode(mode=not self.trainable):
            return torch.cat(
                [self._encode(images=images, encoder=self.encoder, pool=self.pool) for images in data_loader], dim=-1
            )


class WikidataVisualRepresentation(BackfillRepresentation):
    """
    Visual representations obtained from Wikidata and encoded with a vision encoder.

    If no image could be found for a certain Wikidata ID, a plain (trainable) embedding will be used instead.

    Example usage::

    .. code-block:: python

        from pykeen.datasets import get_dataset
        from pykeen.models import ERModel
        from pykeen.nn import WikidataVisualRepresentation
        from pykeen.pipeline import pipeline

        dataset = get_dataset(dataset="codexsmall")
        entity_representations = WikidataVisualRepresentation.from_dataset(dataset=dataset)

        result = pipeline(
            dataset=dataset,
            model=ERModel,
            model_kwargs=dict(
                interaction="distmult",
                entity_representations=entity_representations,
                relation_representation_kwargs=dict(
                    shape=entity_representations.shape,
                ),
            ),
        )
    """

    def __init__(
        self, wikidata_ids: Sequence[str], max_id: Optional[int] = None, image_kwargs: OptionalKwargs = None, **kwargs
    ):
        """
        Initialize the representation.

        :param wikidata_ids:
            the Wikidata IDs
        :param max_id:
            the total number of IDs. If provided, must match the length of `wikidata_ids`
        :param image_kwargs:
            keyword-based parameters passed to :meth:`WikidataCache.get_image_paths`
        :param kwargs:
            additional keyword-based parameters passed to :meth:`VisualRepresentation.__init__`

        :raises ValueError:
            if the max_id does not match the number of Wikidata IDs
        """
        max_id = max_id or len(wikidata_ids)
        if len(wikidata_ids) != max_id:
            raise ValueError(f"Inconsistent max_id={max_id} vs. len(wikidata_ids)={len(wikidata_ids)}")
        images = WikidataCache().get_image_paths(wikidata_ids, **(image_kwargs or {}))
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
    ) -> "WikidataVisualRepresentation":
        """
        Prepare a visual representations for Wikidata entities from a triples factory.

        :param triples_factory:
            the triples factory
        :param for_entities:
            whether to create the initializer for entities (or relations)
        :param kwargs:
            additional keyword-based arguments passed to :meth:`WikidataVisualRepresentation.__init__`

        :returns:
            a visual representation from the triples factory
        """
        return cls(
            wikidata_ids=(
                triples_factory.entity_labeling if for_entities else triples_factory.relation_labeling
            ).all_labels(),
            **kwargs,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        **kwargs,
    ) -> "WikidataVisualRepresentation":
        """Prepare representations from a dataset.

        :param dataset:
            the dataset; needs to have Wikidata IDs as entity names
        :param kwargs:
            additional keyword-based parameters passed to
            :meth:`WikidataVisualRepresentation.from_triples_factory`

        :return:
            the representation

        :raises TypeError:
            if the triples factory does not provide labels
        """
        if not isinstance(dataset.training, TriplesFactory):
            raise TypeError(f"{cls.__name__} requires access to labels, but dataset.training does not provide such.")
        return cls.from_triples_factory(triples_factory=dataset.training, **kwargs)
