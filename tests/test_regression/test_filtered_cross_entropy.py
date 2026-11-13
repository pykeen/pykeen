"""Regression tests for label smoothing with filtered sLCWA negatives."""

import pytest
import torch
from torch.nn import functional

from pykeen.losses import CrossEntropyLoss, InfoNCELoss


@pytest.mark.parametrize("loss_cls", [CrossEntropyLoss, InfoNCELoss])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("label_smoothing", [None, 0.0, 0.1, 1.0])
@pytest.mark.parametrize("filtering", ["none", "all_kept", "mixed", "all_removed"])
def test_filtered_label_smoothing(loss_cls, reduction, label_smoothing, filtering):
    """Match values and gradients from cross entropy over each row's retained candidates."""
    loss = loss_cls(reduction=reduction)
    positive_scores = torch.tensor([[1.0], [-0.5], [2.0]], dtype=torch.float64, requires_grad=True)
    negative_scores = torch.tensor(
        [[0.0, 2.0, -1.0, 0.5], [1.5, -2.0, 0.5, 3.0], [-1.0, 0.0, 1.0, 2.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    batch_filter = torch.ones_like(negative_scores, dtype=torch.bool)
    if filtering == "mixed":
        batch_filter[0] = torch.tensor([True, False, True, False])
        batch_filter[1] = False
    elif filtering == "all_removed":
        batch_filter[:] = False

    actual = loss.process_slcwa_scores(
        positive_scores=positive_scores,
        negative_scores=negative_scores if filtering == "none" else negative_scores[batch_filter],
        batch_filter=None if filtering == "none" else batch_filter,
        label_smoothing=label_smoothing,
        num_entities=5,
    )

    reference_positives = positive_scores
    reference_negatives = negative_scores
    if isinstance(loss, InfoNCELoss):
        reference_positives = (reference_positives - loss.margin) / loss.inverse_softmax_temperature
        reference_negatives = reference_negatives / loss.inverse_softmax_temperature
    row_losses = []
    for positive_row, negative_row, keep in zip(reference_positives, reference_negatives, batch_filter, strict=True):
        row_scores = torch.cat([positive_row, negative_row[keep]]).unsqueeze(0)
        row_losses.append(
            functional.cross_entropy(
                row_scores, torch.zeros(1, dtype=torch.long), label_smoothing=label_smoothing or 0.0
            )
        )
    expected = torch.stack(row_losses)
    expected = expected.mean() if reduction == "mean" else expected.sum()

    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(actual, (positive_scores, negative_scores), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, (positive_scores, negative_scores))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, expected_gradient)
