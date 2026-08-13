"""Métricas e calibração para detecção multi-label de Action Units."""
from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from config.settings import AU_NAMES


def _threshold_array(thresholds: float | Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(thresholds, dtype=np.float32)
    if values.ndim == 0:
        values = np.full(len(AU_NAMES), float(values), dtype=np.float32)
    if values.shape != (len(AU_NAMES),):
        raise ValueError(f"Esperados {len(AU_NAMES)} thresholds; recebido {values.shape}")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Thresholds devem estar no intervalo [0, 1]")
    return values


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _icc_3_1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ICC(3,1): concordância de dois avaliadores fixos, medida individual."""
    ratings = np.column_stack((y_true, y_pred)).astype(np.float64)
    n, k = ratings.shape
    if n < 2:
        return 0.0
    row_mean = ratings.mean(axis=1)
    col_mean = ratings.mean(axis=0)
    grand_mean = ratings.mean()
    ms_rows = k * np.square(row_mean - grand_mean).sum() / (n - 1)
    residual = ratings - row_mean[:, None] - col_mean[None, :] + grand_mean
    ms_error = np.square(residual).sum() / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error
    return float((ms_rows - ms_error) / denominator) if denominator > 0 else 0.0


class AUMetrics:
    """Acumula probabilidades e rótulos e calcula métricas por AU."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._binary_true: list[np.ndarray] = []
        self._intensity_pred: list[np.ndarray] = []
        self._intensity_true: list[np.ndarray] = []
        self._probs: list[np.ndarray] = []

    def update(self, predictions: dict, targets: dict, threshold: float = 0.5):
        """Registra um batch. ``threshold`` é aceito por compatibilidade."""
        del threshold
        logits = predictions['binary_logits'].detach().cpu()
        self._probs.append(torch.sigmoid(logits).numpy())
        self._binary_true.append(targets['binary'].detach().cpu().numpy().astype(np.float32))
        self._intensity_pred.append(predictions['intensity'].detach().cpu().numpy())
        self._intensity_true.append(targets['intensity'].detach().cpu().numpy())

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self._probs:
            raise RuntimeError("Nenhuma predição foi acumulada")
        return (
            np.concatenate(self._probs, axis=0),
            np.concatenate(self._binary_true, axis=0),
            np.concatenate(self._intensity_pred, axis=0),
            np.concatenate(self._intensity_true, axis=0),
        )

    def calibrate_thresholds(
        self,
        minimum: float = 0.05,
        maximum: float = 0.95,
        step: float = 0.01,
    ) -> np.ndarray:
        """Escolhe, somente neste conjunto, o threshold que maximiza F1 de cada AU."""
        probs, binary_true, _, _ = self._arrays()
        grid = np.arange(minimum, maximum + step / 2, step)
        thresholds = np.full(len(AU_NAMES), 0.5, dtype=np.float32)

        for i in range(len(AU_NAMES)):
            if binary_true[:, i].sum() == 0:
                continue
            scores = np.array([
                f1_score(binary_true[:, i], probs[:, i] >= value, zero_division=0)
                for value in grid
            ])
            # Em empates, o maior threshold evita favorecer falsos positivos.
            thresholds[i] = float(grid[np.flatnonzero(scores == scores.max())[-1]])
        return thresholds

    def compute(self, thresholds: float | Sequence[float] | np.ndarray = 0.5) -> dict:
        probs, binary_true, intens_pred, intens_true = self._arrays()
        threshold_values = _threshold_array(thresholds)
        binary_pred = (probs >= threshold_values[None, :]).astype(np.float32)

        precision_per_au = np.array([
            precision_score(binary_true[:, i], binary_pred[:, i], zero_division=0)
            for i in range(len(AU_NAMES))
        ])
        recall_per_au = np.array([
            recall_score(binary_true[:, i], binary_pred[:, i], zero_division=0)
            for i in range(len(AU_NAMES))
        ])
        f1_per_au = np.array([
            f1_score(binary_true[:, i], binary_pred[:, i], zero_division=0)
            for i in range(len(AU_NAMES))
        ])

        ap_per_au = np.array([
            average_precision_score(binary_true[:, i], probs[:, i])
            if binary_true[:, i].sum() > 0 else 0.0
            for i in range(len(AU_NAMES))
        ])

        # A ocorrência decide se a intensidade final é zero. Também preservamos
        # MAE somente em frames ativos para comparar diretamente a regressão bruta.
        gated_intensity = intens_pred * binary_pred
        mae_per_au = np.abs(gated_intensity - intens_true).mean(axis=0)
        rmse_per_au = np.sqrt(np.square(gated_intensity - intens_true).mean(axis=0))
        mae_active_per_au = np.zeros(len(AU_NAMES), dtype=np.float32)
        pearson_per_au = np.zeros(len(AU_NAMES), dtype=np.float32)
        icc_per_au = np.zeros(len(AU_NAMES), dtype=np.float32)
        for i in range(len(AU_NAMES)):
            mask = binary_true[:, i] > 0
            if mask.any():
                mae_active_per_au[i] = np.abs(
                    intens_pred[mask, i] - intens_true[mask, i]
                ).mean()
            pearson_per_au[i] = _pearson(intens_true[:, i], gated_intensity[:, i])
            icc_per_au[i] = _icc_3_1(intens_true[:, i], gated_intensity[:, i])

        report = classification_report(
            binary_true,
            binary_pred,
            target_names=AU_NAMES,
            zero_division=0,
        )

        return {
            'thresholds': threshold_values,
            'precision_per_au': precision_per_au,
            'recall_per_au': recall_per_au,
            'f1_per_au': f1_per_au,
            'precision_macro': float(precision_score(binary_true, binary_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(binary_true, binary_pred, average='macro', zero_division=0)),
            'precision_micro': float(precision_score(binary_true, binary_pred, average='micro', zero_division=0)),
            'recall_micro': float(recall_score(binary_true, binary_pred, average='micro', zero_division=0)),
            'f1_macro': float(f1_score(binary_true, binary_pred, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(binary_true, binary_pred, average='weighted', zero_division=0)),
            'mae_per_au': mae_per_au,
            'mae_mean': float(mae_per_au.mean()),
            'mae_active_per_au': mae_active_per_au,
            'mae_active_mean': float(mae_active_per_au.mean()),
            'rmse_per_au': rmse_per_au,
            'rmse_mean': float(rmse_per_au.mean()),
            'pearson_per_au': pearson_per_au,
            'pearson_mean': float(pearson_per_au.mean()),
            'icc_per_au': icc_per_au,
            'icc_mean': float(icc_per_au.mean()),
            'map': float(ap_per_au.mean()),
            'ap_per_au': ap_per_au,
            'classification_report': report,
        }

    @staticmethod
    def format_summary(results: dict) -> str:
        lines = [
            f"{'AU':<8} {'Prec':>6} {'Rec':>6} {'F1':>6} {'MAE':>6} {'AP':>6} {'Thr':>6}",
            "-" * 55,
        ]
        for i, au in enumerate(AU_NAMES):
            lines.append(
                f"{au:<8} {results['precision_per_au'][i]:>6.4f} "
                f"{results['recall_per_au'][i]:>6.4f} {results['f1_per_au'][i]:>6.4f} "
                f"{results['mae_per_au'][i]:>6.4f} {results['ap_per_au'][i]:>6.4f} "
                f"{results['thresholds'][i]:>6.2f}"
            )
        lines += [
            "-" * 55,
            f"{'Macro Precision':<19} {results['precision_macro']:.4f}",
            f"{'Macro Recall':<19} {results['recall_macro']:.4f}",
            f"{'Macro F1':<19} {results['f1_macro']:.4f}",
            f"{'Weighted F1':<19} {results['f1_weighted']:.4f}",
            f"{'Mean MAE (gated)':<19} {results['mae_mean']:.4f}",
            f"{'Mean MAE (active)':<19} {results['mae_active_mean']:.4f}",
            f"{'Mean RMSE':<19} {results['rmse_mean']:.4f}",
            f"{'Mean Pearson':<19} {results['pearson_mean']:.4f}",
            f"{'Mean ICC(3,1)':<19} {results['icc_mean']:.4f}",
            f"{'mAP':<19} {results['map']:.4f}",
        ]
        return "\n".join(lines)
