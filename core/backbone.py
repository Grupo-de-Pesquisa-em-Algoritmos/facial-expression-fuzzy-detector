import torch
from torch import nn
from blocks.conv import ConvBlock
from blocks.c3k2 import C3K2Block
from blocks.spff import SPFFBlock


class YOLOv11Backbone(nn.Module):
    """
    Backbone do YOLOv11 para extração de features
    
    Esta implementação segue a arquitetura do YOLOv11, com estágios progressivos
    de downsampling e extração de features em múltiplas escalas.
    
    Saídas:
        - P3: features em 1/8 da resolução original (escala pequena)
        - P4: features em 1/16 da resolução original (escala média)
        - P5: features em 1/32 da resolução original (escala grande)
    """
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        
        # Stem: /2
        self.stem = ConvBlock(in_channels, base_channels, kernel_size=3, stride=2, padding=1)
        
        # Stage 1: /4
        self.stage1 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            C3K2Block(base_channels * 2, base_channels * 2, bottleneck_channels=1, shortcut=True)
        )
        
        # Stage 2: /8 (P3)
        self.stage2 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            C3K2Block(base_channels * 4, base_channels * 4, bottleneck_channels=2, shortcut=True)
        )
        
        # Stage 3: /16 (P4)
        self.stage3 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
            C3K2Block(base_channels * 8, base_channels * 8, bottleneck_channels=2, shortcut=True)
        )
        
        # Stage 4: /32 (P5)
        self.stage4 = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16, kernel_size=3, stride=2, padding=1),
            C3K2Block(base_channels * 16, base_channels * 16, bottleneck_channels=1, shortcut=True)
        )
        
        # Stage 5: refina P5 sem reduzir novamente a resolução espacial.
        self.stage5 = nn.Sequential(
            ConvBlock(base_channels * 16, base_channels * 16, kernel_size=3, stride=1, padding=1),
            C3K2Block(base_channels * 16, base_channels * 16, bottleneck_channels=1, shortcut=True),
            SPFFBlock(base_channels * 16, base_channels * 16, kernel_sizes=[5, 9, 13])
        )

    def forward(self, x):
        """
        Forward pass através da backbone
        
        Args:
            x: Tensor de entrada (B, 3, H, W)
            
        Returns:
            Tupla com três tensores de features em diferentes escalas (P3, P4, P5)
        """
        x = self.stem(x)       # /2
        x = self.stage1(x)     # /4

        p3 = self.stage2(x)    # /8  - features de escala pequena
        p4 = self.stage3(p3)   # /16 - features de escala média
        p5 = self.stage4(p4)   # /32 - features de escala grande
        p5 = self.stage5(p5)   # refinamento sem downsampling
        
        return p3, p4, p5
