"""Equivalence tests between grouped and dense sLCWA batch processing."""

import pytest
import torch

from pykeen.datasets import Nations
from pykeen.datasets.base import Dataset
from pykeen.losses import BCEWithLogitsLoss, MarginRankingLoss
from pykeen.models import UM, ComplEx, DistMult, RotatE, TransE
from pykeen.sampling import BasicNegativeSampler, expand_corruption
from pykeen.sampling.filtering import PythonSetFilterer
from pykeen.training.slcwa import SLCWATrainingLoop
from pykeen.triples.instances import GroupedSLCWABatch, SLCWABatch

MODELS = [DistMult, TransE, RotatE, ComplEx, UM]


def _make_batches(*, filtered: bool = False, weighted: bool = False) -> tuple[GroupedSLCWABatch, SLCWABatch, Dataset]:
    dataset = Nations()
    mapped_triples = dataset.training.mapped_triples
    positives = mapped_triples[:8]

    sampler_kwargs = {
        "mapped_triples": mapped_triples,
        "num_negs_per_pos": 6,
        "corruption_scheme": ("head", "tail"),
    }
    if filtered:
        sampler_kwargs["filterer"] = PythonSetFilterer
    sampler = BasicNegativeSampler(**sampler_kwargs)

    corruptions, masks = sampler.sample_grouped(positive_batch=positives)
    grouped_batch: GroupedSLCWABatch = {"positives": positives, "corruptions": corruptions}
    if masks is not None:
        grouped_batch["masks"] = masks

    negatives = torch.cat(
        [expand_corruption(positives, target, replacements) for target, replacements in corruptions.items()], dim=1
    )
    dense_batch: SLCWABatch = {"positives": positives, "negatives": negatives}
    if masks is not None:
        dense_batch["masks"] = torch.cat([masks[target] for target in corruptions], dim=1)

    if weighted:
        h, r, t = positives.unbind(dim=-1)
        pos_weights = (h + r + t + 1).float().unsqueeze(dim=-1)
        grouped_batch["pos_weights"] = pos_weights
        dense_batch["pos_weights"] = pos_weights
        grouped_neg_weights = {}
        dense_neg_weights_list = []
        for target, replacements in corruptions.items():
            if target == "head":
                weights = (replacements + r[:, None] + t[:, None] + 1).float()
            else:
                weights = (h[:, None] + r[:, None] + replacements + 1).float()
            grouped_neg_weights[target] = weights
            dense_neg_weights_list.append(weights)
        grouped_batch["neg_weights"] = grouped_neg_weights
        dense_batch["neg_weights"] = torch.cat(dense_neg_weights_list, dim=1)

    return grouped_batch, dense_batch, dataset


@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_grouped_dense_equivalence(model_cls, filtered: bool, weighted: bool):
    """Test that grouped and dense sLCWA batch processing yield the same loss for the same corruptions."""
    if filtered and weighted:
        pytest.skip("dense path does not mask neg_weights; not comparable when both filtered and weighted")

    torch.manual_seed(0)
    grouped_batch, dense_batch, dataset = _make_batches(filtered=filtered, weighted=weighted)

    model = model_cls(triples_factory=dataset.training, random_seed=0)
    model.eval()  # neutralise dropout/BatchNorm
    loss = BCEWithLogitsLoss() if weighted else MarginRankingLoss()

    loss_grouped = SLCWATrainingLoop._process_batch_static(
        model=model, loss=loss, mode=None, batch=grouped_batch, start=None, stop=None
    )
    loss_dense = SLCWATrainingLoop._process_batch_static(
        model=model, loss=loss, mode=None, batch=dense_batch, start=None, stop=None
    )
    torch.testing.assert_close(loss_grouped, loss_dense)
