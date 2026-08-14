"""
Core components do YOLOv11 para detecção de Action Units (FACS)
"""
from .backbone import YOLOv11Backbone
from .neck import YOLOv11Neck
from .head import AUDetectionHead, YOLOv11AUDetector
from .roi_head import ROIAlignAUHead

__all__ = [
    'YOLOv11Backbone',
    'YOLOv11Neck',
    'AUDetectionHead',
    'YOLOv11AUDetector',
    'ROIAlignAUHead',
]

