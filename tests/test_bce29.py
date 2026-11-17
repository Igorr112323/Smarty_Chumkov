import math

import pytest

torch = pytest.importorskip("torch")

from tools.train_bce29 import weighted_loss


def test_weighted_loss_uses_one_logit_and_ignores_padding():
    logits = torch.zeros((1, 4, 2), dtype=torch.float)
    labels = torch.tensor([[0, 1, 0, -100]])
    loss, count = weighted_loss(torch, logits, labels)
    assert count == 3
    assert float(loss) == pytest.approx(math.log(2) * 13 / 3)


def test_weighted_loss_follows_bce_with_logits():
    logits = torch.tensor([[[0.3, -0.7], [-0.2, 0.5]]])
    labels = torch.tensor([[0, 1]])
    loss, count = weighted_loss(torch, logits, labels)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([-1.0, 0.7]), torch.tensor([0.0, 1.0]), pos_weight=torch.tensor([11.0]))
    assert count == 2
    assert float(loss) == pytest.approx(float(expected))


def test_weighted_loss_rejects_no_labels():
    with pytest.raises(ValueError, match="нет размеченных"):
        weighted_loss(torch, torch.zeros((1, 1, 2)), torch.tensor([[-100]]))
