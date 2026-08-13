"""
dataset_loader.py — Dataset DISFA+ para detecção de Action Units.

Estrutura esperada em disco:
    datasets/archive/
    ├── Images/
    │   └── SN001/SN001/<sessão>/<frame>.jpg
    └── Labels/
        └── SN001/SN001/<sessão>/AUXX.txt

Cada AUXX.txt tem uma linha por frame:
    000.jpg     0
    001.jpg     1
    ...
"""
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config.settings import AU_NAMES, DISFA_DIR, DEFAULT_IMAGE_CONFIG, ImageConfig

# ─── Sujeitos presentes no dataset DISFA+ ────────────────────────────────────
DISFA_SUBJECTS = [
    'SN001', 'SN003', 'SN004', 'SN007', 'SN009',
    'SN010', 'SN013', 'SN025', 'SN027',
]

# Divisão fixa e reprodutível por identidade. A validação serve para escolher
# checkpoint e thresholds; os sujeitos de teste ficam intocados até o relatório final.
DEFAULT_TRAIN_SUBJECTS = DISFA_SUBJECTS[:6]
DEFAULT_VAL_SUBJECTS = DISFA_SUBJECTS[6:7]
DEFAULT_TEST_SUBJECTS = DISFA_SUBJECTS[7:]


def _parse_label_file(path: Path) -> dict[str, float]:
    """
    Lê um arquivo de rótulo AUXX.txt e retorna {frame_stem: intensity}.
    """
    labels: dict[str, float] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r'\s+', line, maxsplit=1)
        if len(parts) != 2:
            continue
        fname, val = parts
        stem = Path(fname).stem  # '000', '001', …
        try:
            labels[stem] = float(val)
        except ValueError:
            continue
    return labels


def _collect_samples(
    images_dir: Path,
    labels_dir: Path,
    subjects: list[str],
    max_samples: Optional[int] = None,
) -> list[dict]:
    """
    Percorre todos os sujeitos/sessões e monta a lista de amostras.

    Cada amostra é:
        {'image_path': Path, 'au_intensities': ndarray(12,)}
    """
    samples: list[dict] = []

    for subject in subjects:
        img_subj_dir = images_dir / subject / subject
        lbl_subj_dir = labels_dir / subject / subject

        if not img_subj_dir.is_dir():
            continue

        for session_dir in sorted(img_subj_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            session = session_dir.name
            lbl_session_dir = lbl_subj_dir / session

            # Carregar intensidades de cada AU (se arquivo existir, senão 0)
            au_labels: dict[str, dict[str, float]] = {}
            for au_name in AU_NAMES:
                lbl_file = lbl_session_dir / f"{au_name}.txt"
                if lbl_file.is_file():
                    au_labels[au_name] = _parse_label_file(lbl_file)
                else:
                    au_labels[au_name] = {}

            # Enumerar imagens da sessão
            for img_path in sorted(session_dir.glob('*.jpg')):
                stem = img_path.stem
                intensities = np.zeros(len(AU_NAMES), dtype=np.float32)
                for idx, au_name in enumerate(AU_NAMES):
                    intensities[idx] = au_labels[au_name].get(stem, 0.0)

                samples.append({
                    'image_path': img_path,
                    'au_intensities': intensities,
                })
                if max_samples is not None and len(samples) >= max_samples:
                    return samples

    return samples


class DisfaDataset(Dataset):
    """
    Dataset DISFA+ para detecção de Action Units.

    Args:
        subjects:   lista de IDs de sujeito (ex: ['SN001', 'SN003']).
                    Se None, usa todos os 9 sujeitos.
        img_config: configuração de imagem (tamanho, normalização).
        augment:    se True, aplica augmentações aleatórias (flip, color jitter).
        disfa_dir:  diretório raiz do dataset (padrão: DISFA_DIR da config).
        max_samples: limita amostras para smoke tests; None usa todas.
    """

    def __init__(
        self,
        subjects: Optional[list[str]] = None,
        img_config: Optional[ImageConfig] = None,
        augment: bool = False,
        disfa_dir: Optional[Path] = None,
        max_samples: Optional[int] = None,
    ):
        self.img_config = img_config or DEFAULT_IMAGE_CONFIG
        self.augment = augment

        root = Path(disfa_dir) if disfa_dir else DISFA_DIR
        images_dir = root / 'Images'
        labels_dir = root / 'Labels'
        used_subjects = DISFA_SUBJECTS if subjects is None else subjects

        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples deve ser maior que zero")

        self.samples = _collect_samples(
            images_dir,
            labels_dir,
            used_subjects,
            max_samples=max_samples,
        )
        if not self.samples:
            raise RuntimeError(
                f"Nenhuma amostra encontrada em {root}. "
                "Verifique a estrutura do dataset DISFA+."
            )

        # Transformações de imagem
        h, w = self.img_config.height, self.img_config.width
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        ] if augment else []

        self.transform = transforms.Compose([
            transforms.Resize((h, w)),
            *aug_transforms,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(self.img_config.mean),
                std=list(self.img_config.std),
            ),
        ])

    # ── Estatísticas para pos_weight BCE ──────────────────────────────────────
    def compute_pos_weight(
        self,
        device: str = 'cpu',
        mode: str = 'sqrt',
        max_weight: Optional[float] = 5.0,
    ) -> torch.Tensor:
        """
        Calcula pos_weight por AU para BCEWithLogitsLoss.

        ``balanced`` usa a razão negativa/positiva completa; ``sqrt`` reduz
        sua agressividade e é o padrão; ``none`` desliga o balanceamento.
        ``max_weight`` evita que AUs muito raras dominem todo o gradiente.
        """
        all_binary = np.stack([
            (s['au_intensities'] > 0).astype(np.float32)
            for s in self.samples
        ])  # (N, 12)
        pos = all_binary.sum(axis=0)
        neg = len(self.samples) - pos
        ratio = neg / (pos + 1e-6)
        if mode == 'balanced':
            pw = ratio
        elif mode == 'sqrt':
            pw = np.sqrt(ratio)
        elif mode == 'none':
            pw = np.ones_like(ratio)
        else:
            raise ValueError("mode deve ser 'none', 'sqrt' ou 'balanced'")

        if max_weight is not None:
            if max_weight <= 0:
                raise ValueError("max_weight deve ser maior que zero")
            pw = np.minimum(pw, max_weight)
        return torch.tensor(pw, dtype=torch.float32, device=device)

    # ── Dataset interface ──────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img = Image.open(sample['image_path']).convert('RGB')
        img_tensor = self.transform(img)

        intensities = torch.tensor(sample['au_intensities'], dtype=torch.float32)
        binary = (intensities > 0).float()

        return {
            'image':     img_tensor,       # (3, H, W)
            'binary':    binary,           # (12,) 0/1
            'intensity': intensities,      # (12,) 0–5
        }


# ─── Factory ─────────────────────────────────────────────────────────────────

def create_dataloaders(
    subjects_train: Optional[list[str]] = None,
    subjects_val: Optional[list[str]] = None,
    img_config: Optional[ImageConfig] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    disfa_dir: Optional[Path] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Cria DataLoaders de treino e validação.

    Por padrão usa seis sujeitos para treino e um sujeito separado para validação.
    Os dois sujeitos restantes são reservados para o teste final.
    max_train_samples/max_val_samples permitem testes rápidos do pipeline.
    """
    subjects_train = DEFAULT_TRAIN_SUBJECTS if subjects_train is None else subjects_train
    subjects_val = DEFAULT_VAL_SUBJECTS if subjects_val is None else subjects_val

    overlap = set(subjects_train) & set(subjects_val)
    if overlap:
        raise ValueError(f"Sujeitos repetidos entre treino e validação: {sorted(overlap)}")

    train_ds = DisfaDataset(
        subjects=subjects_train,
        img_config=img_config,
        augment=True,
        disfa_dir=disfa_dir,
        max_samples=max_train_samples,
    )
    val_ds = DisfaDataset(
        subjects=subjects_val,
        img_config=img_config,
        augment=False,
        disfa_dir=disfa_dir,
        max_samples=max_val_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_eval_dataloader(
    subjects: Optional[list[str]] = None,
    img_config: Optional[ImageConfig] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    disfa_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Cria um DataLoader determinístico, sem augmentation, para val/teste."""
    dataset = DisfaDataset(
        subjects=DEFAULT_TEST_SUBJECTS if subjects is None else subjects,
        img_config=img_config,
        augment=False,
        disfa_dir=disfa_dir,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
