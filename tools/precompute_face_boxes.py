"""Gera uma vez o cache de caixas faciais do DISFA+.

Uso:
    python tools/precompute_face_boxes.py --data-dir datasets/archive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.face_preprocessing import (
    create_haar_face_detector,
    detect_largest_face,
    image_cache_key,
    save_face_box_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser(description='Pré-computa caixas faciais do DISFA+')
    parser.add_argument('--data-dir', type=Path, default=Path('datasets/archive'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--subjects', nargs='+')
    parser.add_argument('--max-samples-per-subject', type=int)
    args = parser.parse_args()

    if args.max_samples_per_subject is not None and args.max_samples_per_subject <= 0:
        parser.error('--max-samples-per-subject deve ser maior que zero')

    root = args.data_dir.resolve()
    images_root = root / 'Images'
    output = args.output or root / 'face_boxes.json'
    subjects = args.subjects or sorted(path.name for path in images_root.iterdir() if path.is_dir())
    paths: list[Path] = []
    for subject in subjects:
        subject_paths = sorted((images_root / subject).rglob('*.jpg'))
        if args.max_samples_per_subject:
            subject_paths = subject_paths[:args.max_samples_per_subject]
        paths.extend(subject_paths)

    detector = create_haar_face_detector()
    detected: dict[Path, tuple[int, int, int, int] | None] = {}
    for image_path in tqdm(paths, desc='Detectando faces'):
        image = cv2.imread(str(image_path))
        box = detect_largest_face(image, detector) if image is not None else None
        detected[image_path] = box

    raw_failures = sum(box is None for box in detected.values())
    sessions: dict[Path, list[Path]] = {}
    for image_path in paths:
        sessions.setdefault(image_path.parent, []).append(image_path)

    # Os frames são vídeos ordenados. Uma falha pontual do Haar recebe a caixa
    # do frame detectado mais próximo, evitando descartar amostras e identidades.
    unresolved = 0
    for session_paths in sessions.values():
        valid_indices = [
            index for index, image_path in enumerate(session_paths)
            if detected[image_path] is not None
        ]
        for index, image_path in enumerate(session_paths):
            if detected[image_path] is not None:
                continue
            if not valid_indices:
                unresolved += 1
                continue
            nearest = min(valid_indices, key=lambda valid: abs(valid - index))
            detected[image_path] = detected[session_paths[nearest]]

    boxes = {
        image_cache_key(image_path, root): detected[image_path]
        for image_path in paths
    }

    save_face_box_cache(output, boxes, root)
    print(f'Cache salvo em: {output}')
    print(
        f'Imagens: {len(paths)} | falhas diretas: {raw_failures} | '
        f'preenchidas temporalmente: {raw_failures - unresolved} | não resolvidas: {unresolved}'
    )
    return 1 if unresolved else 0


if __name__ == '__main__':
    raise SystemExit(main())
