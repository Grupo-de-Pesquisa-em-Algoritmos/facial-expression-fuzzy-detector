"""Detecção e cache de caixas faciais para os crops usados pelo ROIAlign."""
from __future__ import annotations

import json
from pathlib import Path

import cv2


FACE_BOX_CACHE_VERSION = 1


def image_cache_key(image_path: Path, dataset_root: Path) -> str:
    return image_path.resolve().relative_to(dataset_root.resolve()).as_posix()


def load_face_box_cache(path: str | Path) -> dict[str, tuple[int, int, int, int]]:
    cache_path = Path(path)
    payload = json.loads(cache_path.read_text(encoding='utf-8'))
    if payload.get('version') != FACE_BOX_CACHE_VERSION:
        raise ValueError(f'Versão incompatível do cache de faces: {cache_path}')
    return {
        key: tuple(map(int, box))
        for key, box in payload.get('boxes', {}).items()
        if box is not None
    }


def save_face_box_cache(
    path: str | Path,
    boxes: dict[str, tuple[int, int, int, int] | None],
    dataset_root: str | Path,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'version': FACE_BOX_CACHE_VERSION,
        'dataset_root': str(Path(dataset_root).resolve()),
        'format': 'xyxy_pixels',
        'boxes': boxes,
    }
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def create_haar_face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f'Não foi possível carregar {cascade_path}')
    return detector


def detect_largest_face(
    image_bgr,
    detector: cv2.CascadeClassifier,
) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detections = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(detections) == 0:
        detections = detector.detectMultiScale(
            cv2.equalizeHist(gray),
            scaleFactor=1.08,
            minNeighbors=3,
            minSize=(50, 50),
        )
    if len(detections) == 0:
        return None
    x, y, width, height = max(detections, key=lambda box: int(box[2]) * int(box[3]))
    return int(x), int(y), int(x + width), int(y + height)


def expand_face_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    margin: float = 0.20,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    margin_x, margin_y = round(width * margin), round(height * margin)
    return (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(image_width, x2 + margin_x),
        min(image_height, y2 + margin_y),
    )
