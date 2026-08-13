"""
evaluator.py — Avalia um modelo AU em um DataLoader e persiste os resultados.
"""
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader

from config.settings import AU_NAMES
from utils.metrics import AUMetrics


class Evaluator:
    """
    Avalia um modelo de Action Units num DataLoader.

    Args:
        model:      instância do YOLOv11AUDetector (ou compatível)
        device:     'cuda' ou 'cpu'
        results_dir: diretório onde salvar os relatórios
    """

    def __init__(self, model: torch.nn.Module, device: str = 'cpu',
                 results_dir: str | Path = 'results'):
        self.model = model
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _collect(self, loader: DataLoader) -> AUMetrics:
        """Executa a inferência uma vez e devolve as predições acumuladas."""
        self.model.eval()
        self.model.to(self.device)
        metrics = AUMetrics()

        for batch in loader:
            images = batch['image'].to(self.device)
            predictions = self.model(images)
            metrics.update(predictions, {
                'binary': batch['binary'],
                'intensity': batch['intensity'],
            })
        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        thresholds: float | Sequence[float] | np.ndarray = 0.5,
    ) -> dict:
        """
        Roda inferência em todos os batches do loader e calcula métricas.

        Args:
            loader:    DataLoader com batches {'image', 'binary', 'intensity'}
            thresholds: limiar global ou um limiar para cada AU

        Returns:
            dict de métricas (ver AUMetrics.compute())
        """
        return self._collect(loader).compute(thresholds=thresholds)

    @torch.no_grad()
    def calibrate_and_save(
        self,
        loader: DataLoader,
        filename: str = 'thresholds.json',
        subjects: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Calibra thresholds por AU na validação e os persiste em JSON."""
        metrics = self._collect(loader)
        thresholds = metrics.calibrate_thresholds()
        results = metrics.compute(thresholds=thresholds)
        payload = {
            'method': 'max_f1_per_au',
            'subjects': list(subjects or []),
            'thresholds': {
                au: float(thresholds[i])
                for i, au in enumerate(AU_NAMES)
            },
        }
        output_path = self.results_dir / filename
        output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f"Thresholds salvos em: {output_path}")
        return thresholds, results

    def evaluate_and_save(
        self,
        loader: DataLoader,
        filename: str = 'evaluation_report.txt',
        thresholds: float | Sequence[float] | np.ndarray = 0.5,
    ) -> dict:
        """
        Avalia e grava um relatório em texto no diretório de resultados.

        Returns:
            dict de métricas
        """
        results = self.evaluate(loader, thresholds=thresholds)

        report_path = self.results_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Action Unit Detection — Evaluation Report ===\n\n")
            f.write(AUMetrics.format_summary(results))
            f.write("\n\n--- Detailed Classification Report ---\n")
            f.write(results['classification_report'])

        print(f"Relatório salvo em: {report_path}")
        return results
