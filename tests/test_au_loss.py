"""
Teste unitário — AULoss
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from utils.au_loss import AULoss
from config.settings import AU_MAX_INTENSITY, NUM_AUS


def _fake_batch(B: int = 4):
    predictions = {
        "binary_logits": torch.randn(B, NUM_AUS),
        "intensity":     torch.rand(B, NUM_AUS) * AU_MAX_INTENSITY,
    }
    targets = {
        "binary":    torch.randint(0, 2, (B, NUM_AUS)).float(),
        "intensity": torch.rand(B, NUM_AUS) * AU_MAX_INTENSITY,
    }
    return predictions, targets


def test_loss_keys():
    criterion = AULoss()
    preds, targets = _fake_batch()
    out = criterion(preds, targets)
    assert "loss" in out
    assert "binary_loss" in out
    assert "intensity_loss" in out


def test_loss_is_scalar():
    criterion = AULoss()
    preds, targets = _fake_batch()
    out = criterion(preds, targets)
    assert out["loss"].dim() == 0


def test_loss_is_positive():
    criterion = AULoss()
    preds, targets = _fake_batch()
    out = criterion(preds, targets)
    assert out["loss"].item() >= 0.0


def test_no_active_aus():
    """Quando não há AU ativa, intensity_loss deve ser 0."""
    criterion = AULoss()
    B = 4
    preds = {
        "binary_logits": torch.randn(B, NUM_AUS),
        "intensity":     torch.rand(B, NUM_AUS) * AU_MAX_INTENSITY,
    }
    targets = {
        "binary":    torch.zeros(B, NUM_AUS),
        "intensity": torch.zeros(B, NUM_AUS),
    }
    out = criterion(preds, targets)
    assert out["intensity_loss"].item() == pytest.approx(0.0)


def test_pos_weight_accepted():
    pw = torch.ones(NUM_AUS) * 5
    criterion = AULoss(pos_weight=pw)
    preds, targets = _fake_batch()
    out = criterion(preds, targets)
    assert out["loss"].item() >= 0.0
