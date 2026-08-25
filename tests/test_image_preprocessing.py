"""Testes do contrato RGB/grayscale compartilhado pelo pipeline."""

import numpy as np
import pytest
from PIL import Image

from config.settings import ImageConfig, get_image_config
from utils.image_preprocessing import build_image_transform


def test_grayscale_converts_rgb_uint8_to_single_luminance_channel():
    red = Image.fromarray(np.full((16, 16, 3), (255, 0, 0), dtype=np.uint8))
    tensor = build_image_transform(get_image_config('grayscale'))(red)
    # PIL/torchvision converte vermelho puro para luminância 8-bit próxima de 76.
    expected = (76 / 255 - 0.5) / 0.5
    assert tensor.shape == (1, 224, 224)
    assert tensor.mean().item() == pytest.approx(expected, abs=0.01)


def test_rgb_contract_remains_backward_compatible():
    tensor = build_image_transform(get_image_config('rgb'))(
        Image.new('RGB', (16, 16), color=(20, 40, 60))
    )
    assert tensor.shape == (3, 224, 224)


def test_image_config_rejects_channel_mode_mismatch():
    with pytest.raises(ValueError, match='grayscale exige channels=1'):
        ImageConfig(width=224, height=224, color_mode='grayscale', channels=3)
