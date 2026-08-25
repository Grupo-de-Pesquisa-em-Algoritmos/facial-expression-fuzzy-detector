"""
Teste unitário — DisfaDataset

Usa diretório temporário com estrutura mínima de DISFA+.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch
from PIL import Image

from utils.dataset_loader import (
    DEFAULT_TEST_SUBJECTS,
    DEFAULT_TRAIN_SUBJECTS,
    DEFAULT_VAL_SUBJECTS,
    DISFA_SUBJECTS,
    DisfaDataset,
)
from config.settings import AU_NAMES, get_image_config
from utils.face_preprocessing import expand_face_box, load_face_box_cache, save_face_box_cache


def _build_fake_disfa(tmp_path: Path, n_frames: int = 5) -> Path:
    subject = "SN001"
    session = "Session1"
    img_dir = tmp_path / "Images" / subject / subject / session
    lbl_dir = tmp_path / "Labels" / subject / subject / session
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    for i in range(n_frames):
        pixels = np.zeros((200, 200, 3), dtype=np.uint8)
        pixels[50:150, 50:150] = 255
        img = Image.fromarray(pixels)
        img.save(img_dir / f"{i:03d}.jpg")

    for au in AU_NAMES:
        lines = [
            f"{i:03d}.jpg\t{5 if i == n_frames - 1 else i % 4}\n"
            for i in range(n_frames)
        ]
        (lbl_dir / f"{au}.txt").write_text("".join(lines))

    return tmp_path


def _write_fake_face_cache(root: Path, n_frames: int = 5) -> Path:
    boxes = {
        f"Images/SN001/SN001/Session1/{i:03d}.jpg": (50, 50, 150, 150)
        for i in range(n_frames)
    }
    path = root / 'face_boxes.json'
    save_face_box_cache(path, boxes, root)
    return path


@pytest.fixture
def fake_disfa(tmp_path):
    return _build_fake_disfa(tmp_path)


def test_dataset_len(fake_disfa):
    ds = DisfaDataset(subjects=["SN001"], disfa_dir=fake_disfa)
    assert len(ds) == 5


def test_dataset_max_samples(fake_disfa):
    ds = DisfaDataset(
        subjects=["SN001"],
        disfa_dir=fake_disfa,
        max_samples=2,
    )
    assert len(ds) == 2


def test_item_shapes(fake_disfa):
    ds = DisfaDataset(subjects=["SN001"], disfa_dir=fake_disfa)
    item = ds[0]
    assert item["image"].shape == (3, 224, 224)
    assert item["binary"].shape == (len(AU_NAMES),)
    assert item["intensity"].shape == (len(AU_NAMES),)


def test_grayscale_dataset_has_one_normalized_channel(fake_disfa):
    ds = DisfaDataset(
        subjects=["SN001"],
        disfa_dir=fake_disfa,
        img_config=get_image_config('grayscale'),
    )
    image = ds[0]["image"]
    assert image.shape == (1, 224, 224)
    assert image.dtype == torch.float32
    assert image.min().item() >= -1.0
    assert image.max().item() <= 1.0


def test_dataset_applies_cached_face_crop(fake_disfa):
    cache = _write_fake_face_cache(fake_disfa)
    full = DisfaDataset(subjects=["SN001"], disfa_dir=fake_disfa)[0]["image"]
    cropped = DisfaDataset(
        subjects=["SN001"],
        disfa_dir=fake_disfa,
        crop_faces=True,
        face_boxes_file=cache,
        face_margin=0.0,
    )[0]["image"]
    assert cropped.shape == full.shape
    assert cropped.mean() > full.mean()


def test_face_crop_requires_complete_cache(fake_disfa):
    cache = _write_fake_face_cache(fake_disfa, n_frames=1)
    with pytest.raises(RuntimeError, match='não contém 4'):
        DisfaDataset(
            subjects=["SN001"],
            disfa_dir=fake_disfa,
            crop_faces=True,
            face_boxes_file=cache,
        )


def test_face_box_cache_round_trip_and_margin_clamping(fake_disfa):
    cache = _write_fake_face_cache(fake_disfa)
    boxes = load_face_box_cache(cache)
    assert boxes['Images/SN001/SN001/Session1/000.jpg'] == (50, 50, 150, 150)
    assert expand_face_box((10, 10, 190, 190), 200, 200, margin=0.2) == (0, 0, 200, 200)


def test_binary_is_indicator(fake_disfa):
    ds = DisfaDataset(subjects=["SN001"], disfa_dir=fake_disfa)
    for i in range(len(ds)):
        item = ds[i]
        expected = (item["intensity"] > 0).float()
        assert torch.allclose(item["binary"], expected)


def test_dataset_preserves_facs_level_five(fake_disfa):
    ds = DisfaDataset(subjects=["SN001"], disfa_dir=fake_disfa)
    assert ds[-1]["intensity"].max().item() == 5.0


def test_pos_weight_shape(fake_disfa):
    ds = DisfaDataset(subjects=["SN001"], disfa_dir=fake_disfa)
    pw = ds.compute_pos_weight()
    assert pw.shape == (len(AU_NAMES),)
    assert (pw >= 0).all()


def test_default_subject_splits_are_disjoint_and_complete():
    splits = [set(DEFAULT_TRAIN_SUBJECTS), set(DEFAULT_VAL_SUBJECTS), set(DEFAULT_TEST_SUBJECTS)]
    assert splits[0].isdisjoint(splits[1])
    assert splits[0].isdisjoint(splits[2])
    assert splits[1].isdisjoint(splits[2])
    assert set().union(*splits) == set(DISFA_SUBJECTS)


def test_sqrt_pos_weight_is_less_aggressive(fake_disfa):
    ds = DisfaDataset(subjects=["SN001"], disfa_dir=fake_disfa)
    balanced = ds.compute_pos_weight(mode='balanced', max_weight=None)
    softened = ds.compute_pos_weight(mode='sqrt', max_weight=None)
    assert torch.allclose(softened.square(), balanced, atol=1e-5)


def test_pos_weight_cap(fake_disfa):
    ds = DisfaDataset(subjects=["SN001"], disfa_dir=fake_disfa)
    pw = ds.compute_pos_weight(mode='balanced', max_weight=1.5)
    assert (pw <= 1.5).all()
