"""Tests for training with batch-local closed world assumption."""

from typing import cast

import pytest
import torch

from pykeen.datasets import Nations
from pykeen.losses import BCEWithLogitsLoss, CrossEntropyLoss, Loss
from pykeen.models import DistMult, TransE
from pykeen.models.mocks import FixedModel
from pykeen.training import bcwa
from pykeen.triples.instances import BatchCWABatch
from pykeen.triples.weights import RelationLossWeighter
from pykeen.typing import LongTensor


@pytest.fixture
def generator() -> torch.Generator:
    """Build a generator with fixed seed for reproducible tests."""
    return torch.manual_seed(seed=42)


def _make_unique(max_id: int, num: int, generator: torch.Generator) -> LongTensor:
    ind = torch.randperm(max_id, generator=generator)[:num]
    ind = ind.unique()
    assert len(ind) == num
    return ind


@pytest.mark.parametrize(
    ("num_unique_heads", "num_unique_relations", "num_unique_tails", "num_triples", "num_entities", "num_relations"),
    [(1, 1, 1, 1, 13, 7), (3, 2, 2, 3, 13, 7)],
)
def test_convert_to_batch_local(
    num_unique_heads: int,
    num_unique_relations: int,
    num_unique_tails: int,
    num_triples: int,
    num_entities: int,
    num_relations: int,
    generator: torch.Generator,
) -> None:
    """Test conversion of batch to local indices."""
    # verify valid test input
    assert num_unique_heads <= num_entities
    assert num_unique_relations <= num_relations
    assert num_unique_tails <= num_entities
    assert num_triples >= max(num_unique_heads, num_unique_relations, num_unique_tails)
    hs = _make_unique(max_id=num_entities, num=num_unique_heads, generator=generator)
    rs = _make_unique(max_id=num_relations, num=num_unique_relations, generator=generator)
    ts = _make_unique(max_id=num_entities, num=num_unique_tails, generator=generator)
    indices_l = []
    for num_unique in (num_unique_heads, num_unique_relations, num_unique_tails):
        index = torch.randint(0, num_unique, size=(num_triples,), generator=generator)
        # ensure that each index occurs at least once
        index[torch.randperm(num_triples)[:num_unique]] = torch.arange(num_unique)
        indices_l.append(index)
    indices_t = torch.stack(indices_l, dim=-1)
    mapped_triples = torch.stack([source[index] for source, index in zip((hs, rs, ts), indices_l, strict=True)], dim=-1)
    (hs_u, rs_u, ts_u), local_triples = bcwa._convert_to_batch_local(mapped_triples)
    assert torch.equal(indices_t, local_triples)
    for i, i_unique in zip((hs, rs, ts), (hs_u, rs_u, ts_u), strict=True):
        assert torch.equal(i.unique(), i_unique)


def _scan_triples(mapped_triples: LongTensor, hs: LongTensor, rs: LongTensor, ts: LongTensor) -> LongTensor:
    """Collect all triples that solely contain the given head/relation/tail indices by a full scan."""
    mask = torch.ones(len(mapped_triples), dtype=torch.bool)
    for i, test_elements in enumerate((hs, rs, ts)):
        mask &= torch.isin(elements=mapped_triples[:, i], test_elements=test_elements)
    return mapped_triples[mask]


def _sorted_rows(x: LongTensor) -> LongTensor:
    return x.unique(dim=0)


def test_head_index(generator: torch.Generator) -> None:
    """Test that the head index finds the same triples as a full scan."""
    mapped_triples = Nations().training.mapped_triples
    index = bcwa._HeadIndex(mapped_triples=mapped_triples)
    for _ in range(5):
        batch = mapped_triples[torch.randperm(len(mapped_triples), generator=generator)[:32]]
        hs, rs, ts = (batch[:, i].unique() for i in range(3))
        expected = _scan_triples(mapped_triples, hs=hs, rs=rs, ts=ts)
        found = index.find(hs=hs, rs=rs, ts=ts)
        assert len(found) == len(expected)
        assert torch.equal(_sorted_rows(found), _sorted_rows(expected))


def test_collator_weights() -> None:
    """Test that the collator produces dense weights for all scored triples."""
    mapped_triples = Nations().training.mapped_triples
    weighter = RelationLossWeighter.inverse_relation_frequency(mapped_triples=mapped_triples)
    collator = bcwa.BatchCWACollator(mapped_triples=mapped_triples, loss_weighter=weighter)
    dataset = bcwa.BatchCWADataset(mapped_triples=mapped_triples)
    batch = collator([dataset[i] for i in range(16)])
    assert "weights" in batch
    assert batch["weights"].shape == (len(batch["heads"]), len(batch["relations"]), len(batch["tails"]))
    assert torch.allclose(
        batch["weights"], weighter.weights[batch["relations"]].view(1, -1, 1).expand_as(batch["weights"])
    )
    # the positive targets are training triples
    positives = torch.stack(
        [
            batch["heads"][batch["positives"][:, 0]],
            batch["relations"][batch["positives"][:, 1]],
            batch["tails"][batch["positives"][:, 2]],
        ],
        dim=-1,
    )
    assert (positives[:, None, :] == mapped_triples[None, :, :]).all(dim=-1).any(dim=-1).all()


def test_inverse_triples() -> None:
    """Test that the data loader uses internal relation IDs, including the inverse relations."""
    triples_factory = Nations(create_inverse_triples=True).training
    model = DistMult(triples_factory=triples_factory)
    loop = bcwa.BatchCWATrainingLoop(model=model, triples_factory=triples_factory)
    loader = loop._create_training_data_loader(
        triples_factory, sampler=None, batch_size=len(triples_factory.mapped_triples) * 2, drop_last=False
    )
    (batch,) = list(loader)
    # full batch -> all (internal) relations occur, i.e., both, the forward and the inverse ones
    assert torch.equal(batch["relations"], torch.arange(model.num_relations))


@pytest.mark.parametrize("loss_cls", [BCEWithLogitsLoss, CrossEntropyLoss])
def test_process_bcwa_scores(loss_cls: type[Loss], generator: torch.Generator) -> None:
    """Test that BCWA scores are processed like LCWA scores for each (head, relation)-pair."""
    loss = loss_cls()
    predictions = torch.rand(3, 2, 5, generator=generator)
    targets = torch.as_tensor([[0, 0, 1], [0, 0, 3], [2, 1, 4]])
    labels = torch.zeros_like(predictions)
    labels[0, 0, 1] = labels[0, 0, 3] = labels[2, 1, 4] = 1.0
    predictions_2d, labels_2d = predictions.view(-1, 5), labels.view(-1, 5)
    if not loss.bcwa_keep_rows_without_positives:
        mask = labels_2d.any(dim=-1)
        predictions_2d, labels_2d = predictions_2d[mask], labels_2d[mask]
    expected = loss.process_lcwa_scores(predictions=predictions_2d, labels=labels_2d)
    assert torch.allclose(loss.process_bcwa_scores(predictions=predictions, positives=targets), expected)


def _batch_gradients(
    loop: bcwa.BatchCWATrainingLoop, batch: BatchCWABatch, slice_size: int | None
) -> dict[str, torch.Tensor]:
    loop.model.zero_grad()
    loop._forward_pass(batch, 0, 1, 1, label_smoothing=0.0, slice_size=slice_size)
    return {name: p.grad.clone() for name, p in loop.model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("loss_cls", [BCEWithLogitsLoss, CrossEntropyLoss])
@pytest.mark.parametrize("target", ["head", "relation", "tail"])
def test_slicing(target: str, loss_cls: type[Loss]) -> None:
    """Test that slicing does not change the gradients."""
    triples_factory = Nations().training
    model = TransE(triples_factory=triples_factory, loss=loss_cls(), random_seed=0)
    loop = bcwa.BatchCWATrainingLoop(model=model, triples_factory=triples_factory, target=target)
    # note: shuffle, since the triples are sorted by head, i.e., the first batch would only contain a single head
    loader = loop._create_training_data_loader(
        triples_factory,
        sampler=None,
        batch_size=32,
        drop_last=False,
        shuffle=True,
        generator=torch.manual_seed(0),
    )
    batch = next(iter(loader))
    expected = _batch_gradients(loop, batch, slice_size=None)
    actual = _batch_gradients(loop, batch, slice_size=4)
    assert expected.keys() == actual.keys()
    for key, value in expected.items():
        assert torch.allclose(value, actual[key], atol=1.0e-06), key


@pytest.mark.parametrize("target", [0, 1, 2])
def test_target(target: int) -> None:
    """Test the loss for each target against a reference which scores each triple individually."""
    triples_factory = Nations().training
    model = TransE(triples_factory=triples_factory, loss=CrossEntropyLoss(), random_seed=0)
    loop = bcwa.BatchCWATrainingLoop(model=model, triples_factory=triples_factory, target=target)
    loader = loop._create_training_data_loader(triples_factory, sampler=None, batch_size=8, drop_last=False)
    batch = next(iter(loader))
    ids = (batch["heads"], batch["relations"], batch["tails"])
    positives = {tuple(row) for row in batch["positives"].tolist()}
    # rows are all combinations of the non-target positions, columns the target candidates
    first, second = (dim for dim in range(3) if dim != target)
    row_losses = []
    for i in range(len(ids[first])):
        for j in range(len(ids[second])):
            local = []
            for k in range(len(ids[target])):
                triple = [0, 0, 0]
                triple[first], triple[second], triple[target] = i, j, k
                local.append(triple)
            labels = torch.as_tensor([float(tuple(triple) in positives) for triple in local])
            if not labels.any():
                continue
            hrt = torch.as_tensor([[ids[dim][x[dim]] for dim in range(3)] for x in local])
            scores = model.score_hrt(hrt).view(-1)
            row_losses.append(torch.nn.functional.cross_entropy(scores, labels / labels.sum()))
    expected = torch.stack(row_losses).mean()
    actual = loop._process_batch(batch, start=0, stop=loop._get_batch_size(batch))
    assert torch.allclose(actual, expected, atol=1.0e-06)


def test_schlichtkrull_sampler() -> None:
    """Test training with the sub-graph sampler."""
    triples_factory = Nations().training
    model = DistMult(triples_factory=triples_factory)
    loop = bcwa.BatchCWATrainingLoop(model=model, triples_factory=triples_factory)
    losses = loop.train(triples_factory=triples_factory, num_epochs=1, batch_size=64, sampler="schlichtkrull")
    assert len(losses) == 1


def test_requires_er_model() -> None:
    """Test that a model which is not an ERModel is rejected."""
    triples_factory = Nations().training
    with pytest.raises(TypeError):
        bcwa.BatchCWATrainingLoop(model=FixedModel(triples_factory=triples_factory), triples_factory=triples_factory)


def test_unknown_batch_triples() -> None:
    """Test that the collator rejects batches with triples it does not know."""
    mapped_triples = Nations().training.mapped_triples
    collator = bcwa.BatchCWACollator(mapped_triples=mapped_triples[:10])
    dataset = bcwa.BatchCWADataset(mapped_triples=mapped_triples)
    with pytest.raises(bcwa.UnknownBatchTriplesError):
        collator([dataset[i] for i in range(10, 20)])


def test_missing_batch_targets() -> None:
    """Test that the training loop rejects batches without targets."""
    triples_factory = Nations().training
    loop = bcwa.BatchCWATrainingLoop(model=TransE(triples_factory=triples_factory), triples_factory=triples_factory)
    h, r, t = triples_factory.mapped_triples[:3].unbind(dim=-1)
    # a batch without the positives, which are filled by the collator
    batch = cast(BatchCWABatch, {"heads": h, "relations": r, "tails": t})
    with pytest.raises(bcwa.MissingBatchTargetsError):
        loop._process_batch(batch, start=0, stop=3)
