"""
Teste unitário — YOLOv11AUDetector
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from core import YOLOv11AUDetector
from config.settings import AU_MAX_INTENSITY, AU_NAMES, NUM_AUS
from core.regions import ANATOMICAL_REGIONS, AU_REGION_NAMES


@pytest.fixture
def model():
    return YOLOv11AUDetector(in_channels=3, base_channels=16, num_aus=NUM_AUS)


def test_output_shapes(model):
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert "binary_logits" in out and "intensity" in out
    assert out["binary_logits"].shape == (2, NUM_AUS)
    assert out["intensity"].shape == (2, NUM_AUS)


def test_backbone_feature_pyramid_scales(model):
    model.eval()
    with torch.no_grad():
        p3, p4, p5 = model.backbone(torch.randn(1, 3, 224, 224))

    assert p3.shape[1:] == (16 * 4, 28, 28)
    assert p4.shape[1:] == (16 * 8, 14, 14)
    assert p5.shape[1:] == (16 * 16, 7, 7)


def test_forward_accepts_odd_image_size():
    odd_model = YOLOv11AUDetector(
        in_channels=3,
        base_channels=16,
        num_aus=NUM_AUS,
    ).eval()

    with torch.no_grad():
        out = odd_model(torch.randn(2, 3, 225, 225))

    assert out["binary_logits"].shape == (2, NUM_AUS)
    assert out["intensity"].shape == (2, NUM_AUS)


def test_intensity_range(model):
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out["intensity"].min().item() >= 0.0
    assert out["intensity"].max().item() <= AU_MAX_INTENSITY


def test_predict_binary(model):
    x = torch.randn(2, 3, 224, 224)
    pred = model.predict(x)
    assert pred["binary"].dtype == torch.bool
    assert pred["binary"].shape == (2, NUM_AUS)


def test_gradient_flow(model):
    from utils.au_loss import AULoss
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    criterion = AULoss()
    B = x.size(0)
    targets = {
        "binary": torch.randint(0, 2, (B, NUM_AUS)).float(),
        "intensity": torch.rand(B, NUM_AUS) * AU_MAX_INTENSITY,
    }
    losses = criterion(out, targets)
    losses["loss"].backward()
    first_param = next(model.backbone.parameters())
    assert first_param.grad is not None


def test_intensity_sigmoid_keeps_gradient(model):
    from utils.au_loss import AULoss

    predictions = model(torch.randn(2, 3, 224, 224))
    targets = {
        "binary": torch.ones(2, NUM_AUS),
        "intensity": torch.full((2, NUM_AUS), AU_MAX_INTENSITY),
    }
    criterion = AULoss(lambda_binary=0.0, lambda_intensity=1.0)
    criterion(predictions, targets)["loss"].backward()

    grad = model.head.intensity_head.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad).item() > 0


@pytest.fixture
def roi_model():
    return YOLOv11AUDetector(
        in_channels=3,
        base_channels=16,
        num_aus=NUM_AUS,
        architecture='roi',
        roi_channels=32,
        roi_size=3,
    )


def test_roi_model_output_shapes_and_metadata(roi_model):
    out = roi_model(torch.randn(2, 3, 224, 224))
    assert out['binary_logits'].shape == (2, NUM_AUS)
    assert out['intensity'].shape == (2, NUM_AUS)
    assert out['region_boxes'].shape == (len(ANATOMICAL_REGIONS), 4)
    assert roi_model.export_config()['architecture'] == 'roi'


def test_roi_model_intensity_range(roi_model):
    out = roi_model(torch.randn(2, 3, 224, 224))
    assert out['intensity'].min().item() >= 0.0
    assert out['intensity'].max().item() <= AU_MAX_INTENSITY


def test_roi_model_gradient_reaches_backbone_and_local_heads(roi_model):
    from utils.au_loss import AULoss

    output = roi_model(torch.randn(2, 3, 224, 224))
    targets = {
        'binary': torch.randint(0, 2, (2, NUM_AUS)).float(),
        'intensity': torch.rand(2, NUM_AUS) * AU_MAX_INTENSITY,
    }
    AULoss()(output, targets)['loss'].backward()
    assert torch.count_nonzero(next(roi_model.backbone.parameters()).grad).item() > 0
    assert torch.count_nonzero(roi_model.head.au_heads[0][-1].weight.grad).item() > 0


def test_every_au_has_valid_anatomical_regions():
    assert set(AU_REGION_NAMES) == set(AU_NAMES)
    for region_names in AU_REGION_NAMES.values():
        assert region_names
        assert all(name in ANATOMICAL_REGIONS for name in region_names)
