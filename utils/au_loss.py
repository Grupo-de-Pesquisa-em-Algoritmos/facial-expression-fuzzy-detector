"""
au_loss.py — Função de perda para detecção de Action Units (AUs).

Combina:
    - BCEWithLogitsLoss: detectar se a AU está ativa (binary)
    - SmoothL1Loss:      estimar a intensidade contínua (0–5)

Uso:
    criterion = AULoss(pos_weight=dataset.compute_pos_weight(device))
    loss_dict = criterion(predictions, targets)
    loss_dict['loss'].backward()
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AULoss(nn.Module):
    """
    Perda combinada para detecção de Action Units.

    Args:
        pos_weight:        Tensor (num_aus,) para BCEWithLogitsLoss.
                           Compensa o desequilíbrio AU negativo/positivo.
                           Se None, usa peso 1 para todas as AUs.
        lambda_binary:     Peso da perda binária (BCE). Padrão: 1.0
        lambda_intensity:  Peso da perda de intensidade (SmoothL1). Padrão: 0.5
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        lambda_binary: float = 1.0,
        lambda_intensity: float = 0.5,
    ):
        super().__init__()
        self.lambda_binary = lambda_binary
        self.lambda_intensity = lambda_intensity

        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')

    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Args:
            predictions: dict com chaves
                'binary_logits' — (B, num_aus) logits não normalizados
                'intensity'     — (B, num_aus) intensidades [0, 5]
            targets: dict com chaves
                'binary'    — (B, num_aus) labels 0/1
                'intensity' — (B, num_aus) intensidades reais [0, 5]

        Returns:
            dict com:
                'loss'           — escalar total (para .backward())
                'binary_loss'    — contribuição BCE
                'intensity_loss' — contribuição SmoothL1
        """
        binary_loss = self.bce(
            predictions['binary_logits'],
            targets['binary'],
        )

        # SmoothL1 só onde a AU está ativa (máscara = binary target)
        mask = targets['binary'].bool()
        if mask.any():
            intensity_loss = self.smooth_l1(
                predictions['intensity'][mask],
                targets['intensity'][mask],
            )
        else:
            intensity_loss = torch.tensor(0.0, device=predictions['intensity'].device)

        total = (
            self.lambda_binary * binary_loss
            + self.lambda_intensity * intensity_loss
        )

        return {
            'loss':           total,
            'binary_loss':    binary_loss,
            'intensity_loss': intensity_loss,
        }
