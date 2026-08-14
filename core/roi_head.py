"""Cabeça local-global de Action Units baseada em regiões e ROIAlign."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align

from blocks.conv import ConvBlock
from config.settings import AU_MAX_INTENSITY, NUM_AUS
from core.regions import ANATOMICAL_REGIONS, AU_REGION_INDICES, REGION_NAMES


class ROIAlignAUHead(nn.Module):
    """Extrai regiões FACS fixas e prediz presença/intensidade por AU.

    Os três níveis do neck são projetados e fundidos na resolução de N3. O
    ROIAlign preserva estrutura local, enquanto um vetor global pequeno fornece
    contexto para relações entre regiões.
    """

    def __init__(
        self,
        in_channels: list[int],
        num_aus: int = NUM_AUS,
        roi_channels: int = 128,
        roi_size: int = 3,
    ):
        super().__init__()
        if num_aus != len(AU_REGION_INDICES):
            raise ValueError('ROIAlignAUHead requer a ordem padrão das AUs')
        if roi_channels < 16:
            raise ValueError('roi_channels deve ser pelo menos 16')
        if roi_size < 2:
            raise ValueError('roi_size deve ser pelo menos 2')

        self.num_aus = num_aus
        self.roi_channels = roi_channels
        self.roi_size = roi_size
        c3, c4, c5 = in_channels

        self.project_n3 = ConvBlock(c3, roi_channels, kernel_size=1)
        self.project_n4 = ConvBlock(c4, roi_channels, kernel_size=1)
        self.project_n5 = ConvBlock(c5, roi_channels, kernel_size=1)
        self.fuse = ConvBlock(roi_channels * 3, roi_channels, kernel_size=3, padding=1)
        self.region_encoder = nn.Sequential(
            ConvBlock(roi_channels, roi_channels, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )

        feature_dim = roi_channels * 2  # região local + contexto global
        hidden_dim = max(64, roi_channels)
        self.au_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 2),
            )
            for _ in range(num_aus)
        ])

        boxes = [ANATOMICAL_REGIONS[name] for name in REGION_NAMES]
        self.register_buffer(
            'normalized_region_boxes',
            torch.tensor(boxes, dtype=torch.float32),
            persistent=True,
        )

    def _fuse_features(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        n3, n4, n5 = features
        target_size = n3.shape[-2:]
        p3 = self.project_n3(n3)
        p4 = F.interpolate(self.project_n4(n4), size=target_size, mode='bilinear', align_corners=False)
        p5 = F.interpolate(self.project_n5(n5), size=target_size, mode='bilinear', align_corners=False)
        return self.fuse(torch.cat((p3, p4, p5), dim=1))

    def _roi_tensor(self, batch_size: int, height: int, width: int, feature: torch.Tensor):
        boxes = self.normalized_region_boxes.to(device=feature.device, dtype=feature.dtype).clone()
        boxes[:, (0, 2)] *= max(width - 1, 1)
        boxes[:, (1, 3)] *= max(height - 1, 1)
        region_count = boxes.shape[0]
        boxes = boxes.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 4)
        batch_indices = torch.arange(
            batch_size,
            device=feature.device,
            dtype=feature.dtype,
        ).repeat_interleave(region_count).unsqueeze(1)
        return torch.cat((batch_indices, boxes), dim=1)

    def forward(self, features: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        fused = self._fuse_features(features)
        batch_size, _, height, width = fused.shape
        rois = self._roi_tensor(batch_size, height, width, fused)
        pooled = roi_align(
            fused,
            rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        region_features = self.region_encoder(pooled).reshape(batch_size, len(REGION_NAMES), -1)
        global_feature = F.adaptive_avg_pool2d(fused, 1).flatten(1)

        outputs = []
        for au_index, region_indices in enumerate(AU_REGION_INDICES):
            local_feature = region_features[:, region_indices].mean(dim=1)
            outputs.append(self.au_heads[au_index](torch.cat((local_feature, global_feature), dim=1)))
        predictions = torch.stack(outputs, dim=1)  # (B, AUs, binary+intensity)

        return {
            'binary_logits': predictions[..., 0],
            'intensity': AU_MAX_INTENSITY * torch.sigmoid(predictions[..., 1]),
            'region_boxes': self.normalized_region_boxes,
        }
