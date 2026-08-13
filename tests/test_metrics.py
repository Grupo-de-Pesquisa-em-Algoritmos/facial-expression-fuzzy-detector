"""Testes das métricas multi-label e da calibração de thresholds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from config.settings import AU_MAX_INTENSITY, NUM_AUS
from utils.metrics import AUMetrics


def _calibration_case():
    probabilities = torch.tensor([0.10, 0.20, 0.40, 0.60, 0.70, 0.80, 0.90, 0.95])
    probabilities = probabilities[:, None].repeat(1, NUM_AUS)
    binary = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.float32)
    binary = binary[:, None].repeat(1, NUM_AUS)
    intensity = binary * AU_MAX_INTENSITY
    predictions = {
        'binary_logits': torch.logit(probabilities),
        'intensity': intensity.clone(),
    }
    targets = {'binary': binary, 'intensity': intensity}
    return predictions, targets


def test_per_au_calibration_reduces_false_positives():
    predictions, targets = _calibration_case()
    metrics = AUMetrics()
    metrics.update(predictions, targets)

    before = metrics.compute(thresholds=0.5)
    thresholds = metrics.calibrate_thresholds()
    after = metrics.compute(thresholds=thresholds)

    assert (thresholds >= 0.8 - 1e-6).all()
    assert after['precision_macro'] > before['precision_macro']
    assert after['f1_macro'] > before['f1_macro']


def test_metrics_report_intensity_and_concordance():
    predictions, targets = _calibration_case()
    metrics = AUMetrics()
    metrics.update(predictions, targets)
    results = metrics.compute(thresholds=0.8)

    assert results['mae_mean'] == 0.0
    assert results['mae_active_mean'] == 0.0
    assert results['rmse_mean'] == 0.0
    assert results['pearson_mean'] == 1.0
    assert results['icc_mean'] == 1.0


def test_threshold_count_is_validated():
    predictions, targets = _calibration_case()
    metrics = AUMetrics()
    metrics.update(predictions, targets)

    try:
        metrics.compute(thresholds=[0.5, 0.5])
    except ValueError:
        pass
    else:
        raise AssertionError("Era esperado ValueError para quantidade incorreta de thresholds")
