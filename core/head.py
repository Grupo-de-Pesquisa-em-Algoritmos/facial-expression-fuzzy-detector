import torch
from torch import nn
from blocks.conv import ConvBlock
from config.settings import AU_MAX_INTENSITY, NUM_AUS


class AUDetectionHead(nn.Module):
    """
    Head para detecção de Action Units (AUs) faciais.

    Recebe features multi-escala do Neck (N3, N4, N5) e produz
    duas saídas independentes por AU:
        - binary_logits: presença/ausência de cada AU  (B, NUM_AUS)
        - intensity:     intensidade de cada AU 0–5     (B, NUM_AUS)
    """

    def __init__(self, in_channels: list[int], num_aus: int = NUM_AUS):
        super().__init__()
        self.num_aus = num_aus
        c3, c4, c5 = in_channels

        # Camadas de redução para cada escala antes do GAP
        self.reduce_n3 = nn.Sequential(
            ConvBlock(c3, c3 // 2, kernel_size=3, padding=1),
            ConvBlock(c3 // 2, c3 // 4, kernel_size=1),
        )
        self.reduce_n4 = nn.Sequential(
            ConvBlock(c4, c4 // 2, kernel_size=3, padding=1),
            ConvBlock(c4 // 2, c4 // 4, kernel_size=1),
        )
        self.reduce_n5 = nn.Sequential(
            ConvBlock(c5, c5 // 2, kernel_size=3, padding=1),
            ConvBlock(c5 // 2, c5 // 4, kernel_size=1),
        )

        fused_dim = c3 // 4 + c4 // 4 + c5 // 4  # ex: 64 + 128 + 256 = 448

        # Cabeça compartilhada
        self.shared = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
        )

        # Cabeça binária: AU ativa ou não
        self.binary_head = nn.Linear(256, num_aus)

        # Cabeça de intensidade: 0–5 contínuo
        self.intensity_head = nn.Linear(256, num_aus)

    def forward(self, features):
        """
        Args:
            features: tupla (N3, N4, N5) do Neck

        Returns:
            dict com:
                'binary_logits': (B, num_aus) — raw logits para BCE
                'intensity':     (B, num_aus) — valores [0, 5] via sigmoid
        """
        n3, n4, n5 = features

        # Reduzir canais e aplicar GAP em cada escala
        f3 = self.reduce_n3(n3)  # (B, c3//4, H, W)
        f4 = self.reduce_n4(n4)  # (B, c4//4, H, W)
        f5 = self.reduce_n5(n5)  # (B, c5//4, H, W)

        gap = nn.functional.adaptive_avg_pool2d
        f3 = gap(f3, 1).flatten(1)  # (B, c3//4)
        f4 = gap(f4, 1).flatten(1)  # (B, c4//4)
        f5 = gap(f5, 1).flatten(1)  # (B, c5//4)

        # Concatenar representações multi-escala
        x = torch.cat([f3, f4, f5], dim=1)  # (B, fused_dim)

        # Cabeça compartilhada
        x = self.shared(x)  # (B, 256)

        binary_logits = self.binary_head(x)                          # (B, num_aus)
        # Sigmoid mantém toda a saída no intervalo válido sem as regiões de
        # gradiente nulo que um clamp aplicado aos logits criaria.
        intensity = AU_MAX_INTENSITY * torch.sigmoid(self.intensity_head(x))

        return {
            'binary_logits': binary_logits,
            'intensity': intensity,
        }


class YOLOv11AUDetector(nn.Module):
    """
    Modelo completo YOLOv11 para detecção de Action Units.

    Pipeline: Backbone → Neck → AUDetectionHead

    Saída:
        'binary_logits': (B, 12) — aplicar sigmoid para probabilidade
        'intensity':     (B, 12) — intensidade [0, 5]
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_aus: int = NUM_AUS,
        architecture: str = 'global',
        roi_channels: int = 128,
        roi_size: int = 3,
    ):
        super().__init__()
        from core.backbone import YOLOv11Backbone
        from core.neck import YOLOv11Neck
        from core.roi_head import ROIAlignAUHead

        if architecture not in {'global', 'roi'}:
            raise ValueError("architecture deve ser 'global' ou 'roi'")

        self.architecture = architecture
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_aus = num_aus
        self.roi_channels = roi_channels
        self.roi_size = roi_size

        self.backbone = YOLOv11Backbone(in_channels=in_channels, base_channels=base_channels)

        channels = [base_channels * 4, base_channels * 8, base_channels * 16]
        self.neck = YOLOv11Neck(channels=channels)
        if architecture == 'roi':
            self.head = ROIAlignAUHead(
                in_channels=channels,
                num_aus=num_aus,
                roi_channels=roi_channels,
                roi_size=roi_size,
            )
        else:
            self.head = AUDetectionHead(in_channels=channels, num_aus=num_aus)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck((p3, p4, p5))
        return self.head((n3, n4, n5))

    def export_config(self) -> dict:
        """Configuração mínima necessária para reconstruir o checkpoint."""
        return {
            'architecture': self.architecture,
            'in_channels': self.in_channels,
            'base_channels': self.base_channels,
            'num_aus': self.num_aus,
            'roi_channels': self.roi_channels,
            'roi_size': self.roi_size,
        }

    @torch.no_grad()
    def predict(self, x, binary_threshold: float = 0.5):
        """
        Inferência com threshold binário aplicado.

        Returns:
            'binary':    (B, 12) bool  — AU ativa ou não
            'intensity': (B, 12) float — intensidade da AU
        """
        out = self.forward(x)
        return {
            'binary':    torch.sigmoid(out['binary_logits']) >= binary_threshold,
            'intensity': out['intensity'],
        }

