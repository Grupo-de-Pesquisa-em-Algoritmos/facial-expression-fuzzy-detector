"""
inference.py — Preditor de Action Units a partir de imagens completas.

Fluxo:
    1. MediaPipe detecta rosto(s) na imagem completa
    2. Crop + resize para (224, 224)
    3. YOLOv11AUDetector prediz AUs
    4. Retorna dict por AU com {'active': bool, 'intensity': float}

Dependência opcional: mediapipe (pip install mediapipe)
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from config.settings import AU_NAMES, AU_DESCRIPTIONS, DEFAULT_IMAGE_CONFIG


_MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass


# ─── Preprocessamento ─────────────────────────────────────────────────────────

def _build_transform(img_config=None):
    cfg = img_config or DEFAULT_IMAGE_CONFIG
    return transforms.Compose([
        transforms.Resize((cfg.height, cfg.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(cfg.mean), std=list(cfg.std)),
    ])


def _crop_face_pil(image: Image.Image, bbox: tuple[float, float, float, float],
                   margin: float = 0.15) -> Image.Image:
    """
    Recorta a região do rosto, com margem relativa ao tamanho do bounding box.

    Args:
        bbox: (xmin, ymin, xmax, ymax) normalizados [0, 1]
    """
    w, h = image.size
    xmin, ymin, xmax, ymax = bbox
    bw = (xmax - xmin) * w
    bh = (ymax - ymin) * h
    x1 = max(0, (xmin * w) - margin * bw)
    y1 = max(0, (ymin * h) - margin * bh)
    x2 = min(w, (xmax * w) + margin * bw)
    y2 = min(h, (ymax * h) + margin * bh)
    return image.crop((x1, y1, x2, y2))


# ─── Preditor principal ───────────────────────────────────────────────────────

class AUPredictor:
    """
    Predição de Action Units a partir de uma imagem (arquivo ou tensor).

    Quando MediaPipe está disponível, detecta rostos automaticamente.
    Caso contrário, usa a imagem inteira como entrada (útil para imagens
    já recortadas como no DISFA+).

    Args:
        model:          YOLOv11AUDetector (ou compatível) já carregado
        device:         'cuda' ou 'cpu'
        threshold:      limiar para classificação binária (padrão 0.5)
        img_config:     configuração de imagem (tamanho, normalização)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        threshold: float | Sequence[float] = 0.5,
        img_config=None,
    ):
        self.model = model.to(device).eval()
        self.device = device
        thresholds = np.asarray(threshold, dtype=np.float32)
        if thresholds.ndim == 0:
            thresholds = np.full(len(AU_NAMES), float(thresholds), dtype=np.float32)
        if thresholds.shape != (len(AU_NAMES),):
            raise ValueError(f"Esperados {len(AU_NAMES)} thresholds; recebido {thresholds.shape}")
        self.thresholds = thresholds
        self.transform = _build_transform(img_config)

        # MediaPipe face detector (carregado sob demanda)
        self._mp_face = None
        if _MEDIAPIPE_AVAILABLE:
            self._mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1,   # full-range model
                min_detection_confidence=0.5,
            )

    # ── API pública ───────────────────────────────────────────────────────────

    def predict_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
    ) -> list[dict]:
        """
        Prediz AUs para cada rosto detectado na imagem.

        Args:
            image: caminho para arquivo, PIL Image ou ndarray HWC uint8

        Returns:
            Lista de dicts, um por rosto detectado:
            {
                'au_results': {
                    'AU1': {'active': bool, 'intensity': float, 'description': str},
                    ...
                },
                'bbox': (xmin, ymin, xmax, ymax)  # normalizados, ou None se imagem inteira
            }
        """
        pil_img = self._load_pil(image)
        face_crops = self._detect_faces(pil_img)

        results = []
        for crop, bbox in face_crops:
            au_results = self._predict_crop(crop)
            results.append({'au_results': au_results, 'bbox': bbox})

        return results

    def predict_file(self, path: Union[str, Path]) -> list[dict]:
        return self.predict_image(path)

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _load_pil(image) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert('RGB')
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert('RGB')
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        raise TypeError(f"Tipo de imagem não suportado: {type(image)}")

    def _detect_faces(self, pil_img: Image.Image) -> list[tuple[Image.Image, tuple | None]]:
        """
        Detecta rostos com MediaPipe ou devolve a imagem inteira.

        Returns:
            Lista de (crop_pil, bbox) onde bbox é (xmin, ymin, xmax, ymax)
            normalizados, ou None se a imagem foi usada inteira.
        """
        if self._mp_face is None:
            return [(pil_img, None)]

        img_np = np.array(pil_img)
        mp_result = self._mp_face.process(img_np)

        if not mp_result.detections:
            # Nenhum rosto → usa imagem inteira
            return [(pil_img, None)]

        faces = []
        for det in mp_result.detections:
            bbox_mp = det.location_data.relative_bounding_box
            xmin = max(0.0, bbox_mp.xmin)
            ymin = max(0.0, bbox_mp.ymin)
            xmax = min(1.0, xmin + bbox_mp.width)
            ymax = min(1.0, ymin + bbox_mp.height)
            crop = _crop_face_pil(pil_img, (xmin, ymin, xmax, ymax))
            faces.append((crop, (xmin, ymin, xmax, ymax)))

        return faces

    @torch.no_grad()
    def _predict_crop(self, crop: Image.Image) -> dict:
        tensor = self.transform(crop).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        out = self.model(tensor)

        probs     = torch.sigmoid(out['binary_logits']).squeeze(0).cpu().numpy()
        intensity = out['intensity'].squeeze(0).cpu().numpy()

        au_results = {}
        for i, au_name in enumerate(AU_NAMES):
            active = bool(probs[i] >= self.thresholds[i])
            au_results[au_name] = {
                'active':      active,
                'probability': float(probs[i]),
                'intensity':   float(intensity[i]) if active else 0.0,
                'raw_intensity': float(intensity[i]),
                'description': AU_DESCRIPTIONS[au_name],
            }
        return au_results


# ─── Utilitário para salvar resultados ────────────────────────────────────────

def format_au_results(face_results: list[dict]) -> str:
    """Formata a lista de resultados como texto legível."""
    lines = []
    for fi, face in enumerate(face_results):
        lines.append(f"Rosto {fi + 1}:")
        if face['bbox'] is not None:
            x1, y1, x2, y2 = face['bbox']
            lines.append(f"  BBox: ({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})")
        lines.append(f"  {'AU':<8} {'Ativa':<8} {'Prob':>6} {'Intens':>7}  Descrição")
        lines.append("  " + "-" * 55)
        for au_name, info in face['au_results'].items():
            marker = "✓" if info['active'] else " "
            lines.append(
                f"  {au_name:<8} {marker:<8} {info['probability']:>6.3f} "
                f"{info['intensity']:>7.3f}  {info['description']}"
            )
    return "\n".join(lines)
