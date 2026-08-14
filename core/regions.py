"""Definição reprodutível das regiões anatômicas usadas pelo ROIAlign.

As coordenadas estão normalizadas no crop facial (x1, y1, x2, y2).
Elas codificam prior FACS; não são bounding boxes anotadas ou preditas.
"""
from __future__ import annotations

from config.settings import AU_NAMES


ANATOMICAL_REGIONS: dict[str, tuple[float, float, float, float]] = {
    'left_brow':       (0.06, 0.12, 0.50, 0.37),
    'right_brow':      (0.50, 0.12, 0.94, 0.37),
    'central_brow':    (0.24, 0.12, 0.76, 0.39),
    'left_eye_cheek':  (0.05, 0.25, 0.51, 0.64),
    'right_eye_cheek': (0.49, 0.25, 0.95, 0.64),
    'nose':            (0.28, 0.34, 0.72, 0.68),
    'mouth':           (0.14, 0.54, 0.86, 0.86),
    'chin_jaw':        (0.18, 0.66, 0.82, 0.98),
}

REGION_NAMES = tuple(ANATOMICAL_REGIONS)
REGION_INDEX = {name: index for index, name in enumerate(REGION_NAMES)}

AU_REGION_NAMES: dict[str, tuple[str, ...]] = {
    'AU1':  ('left_brow', 'right_brow', 'central_brow'),
    'AU2':  ('left_brow', 'right_brow'),
    'AU4':  ('central_brow',),
    'AU5':  ('left_eye_cheek', 'right_eye_cheek'),
    'AU6':  ('left_eye_cheek', 'right_eye_cheek'),
    'AU9':  ('nose',),
    'AU12': ('mouth',),
    'AU15': ('mouth',),
    'AU17': ('chin_jaw',),
    'AU20': ('mouth',),
    'AU25': ('mouth',),
    'AU26': ('mouth', 'chin_jaw'),
}

AU_REGION_INDICES: tuple[tuple[int, ...], ...] = tuple(
    tuple(REGION_INDEX[name] for name in AU_REGION_NAMES[au])
    for au in AU_NAMES
)


def validate_region_definition() -> None:
    if set(AU_REGION_NAMES) != set(AU_NAMES):
        missing = set(AU_NAMES) - set(AU_REGION_NAMES)
        extra = set(AU_REGION_NAMES) - set(AU_NAMES)
        raise ValueError(f'Mapeamento AU/região inválido; faltando={missing}, extras={extra}')
    for name, (x1, y1, x2, y2) in ANATOMICAL_REGIONS.items():
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError(f'Região {name} fora do intervalo normalizado')


validate_region_definition()
