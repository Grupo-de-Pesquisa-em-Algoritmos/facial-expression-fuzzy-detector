import torch
from torch import nn
from torch.nn import functional as F
from blocks.conv import ConvBlock
from blocks.c3k2 import C3K2Block
from blocks.c2psa import C2PSABlock


class YOLOv11Neck(nn.Module):
    """
    Neck do YOLOv11 com arquitetura PANet (Path Aggregation Network)
    
    Combina features de diferentes escalas da backbone usando:
    - Top-down pathway: propaga informação semântica de alto nível
    - Bottom-up pathway: propaga informação espacial de baixo nível
    
    Entradas:
        - P3, P4, P5: features da backbone em diferentes escalas
        
    Saídas:
        - N3, N4, N5: features refinadas para o head
    """
    def __init__(self, channels=(256, 512, 1024)):
        super().__init__()
        c3, c4, c5 = channels
        
        # Top-down pathway (de P5 para P3)
        self.reduce_c5 = ConvBlock(c5, c4, kernel_size=1)
        self.c3k2_td1 = C3K2Block(c4 + c4, c4, bottleneck_channels=1, shortcut=False)
        
        self.reduce_c4 = ConvBlock(c4, c3, kernel_size=1)
        self.c3k2_td2 = C3K2Block(c3 + c3, c3, bottleneck_channels=1, shortcut=False)
        
        # Bottom-up pathway (de P3 para P5)
        self.downsample1 = ConvBlock(c3, c4, kernel_size=3, stride=2, padding=1)
        self.c3k2_bu1 = C3K2Block(c4 + c4, c4, bottleneck_channels=1, shortcut=False)
        
        self.downsample2 = ConvBlock(c4, c5, kernel_size=3, stride=2, padding=1)
        self.c3k2_bu2 = C3K2Block(c5 + c5, c5, bottleneck_channels=1, shortcut=False)
        
        # C2PSA para refinar features finais
        self.c2psa_n3 = C2PSABlock(c3, c3, bottleneck_channels=1, shortcut=False)
        self.c2psa_n4 = C2PSABlock(c4, c4, bottleneck_channels=1, shortcut=False)
        self.c2psa_n5 = C2PSABlock(c5, c5, bottleneck_channels=1, shortcut=False)

    def forward(self, features):
        """
        Forward pass através do neck
        
        Args:
            features: Tupla (P3, P4, P5) da backbone
            
        Returns:
            Tupla (N3, N4, N5) com features refinadas
        """
        p3, p4, p5 = features
        
        # Top-down pathway
        # P5 -> P4
        x = self.reduce_c5(p5)
        x = F.interpolate(x, size=p4.shape[-2:], mode='nearest')
        x = torch.cat([x, p4], dim=1)
        x = self.c3k2_td1(x)  # features intermediárias em escala P4
        p4_td = x
        
        # P4 -> P3
        x = self.reduce_c4(x)
        x = F.interpolate(x, size=p3.shape[-2:], mode='nearest')
        x = torch.cat([x, p3], dim=1)
        n3 = self.c3k2_td2(x)  # N3 - features de escala pequena
        
        # Bottom-up pathway
        # P3 -> P4
        x = self.downsample1(n3)
        x = F.interpolate(x, size=p4_td.shape[-2:], mode='nearest')
        x = torch.cat([x, p4_td], dim=1)
        n4 = self.c3k2_bu1(x)  # N4 - features de escala média
        
        # P4 -> P5
        x = self.downsample2(n4)
        x = F.interpolate(x, size=p5.shape[-2:], mode='nearest')
        x = torch.cat([x, p5], dim=1)
        n5 = self.c3k2_bu2(x)  # N5 - features de escala grande
        
        # Refinar com C2PSA (atenção espacial)
        n3 = self.c2psa_n3(n3)
        n4 = self.c2psa_n4(n4)
        n5 = self.c2psa_n5(n5)
        
        return n3, n4, n5
