from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_SETS_DIR = ROOT_DIR / "datasets"
DISFA_DIR = DATA_SETS_DIR / "archive"


@dataclass
class ImageConfig:
    width: int
    height: int
    channels: int = 3
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    model_name: str
    input_shape: Tuple[int, int, int]
    num_aus: int
    learning_rate: float
    epochs: int
    batch_size: int


@dataclass
class TrainingConfig:
    train_split: float
    validation_split: float
    shuffle: bool
    accumulation_steps: int = 4
    use_amp: bool = True
    augmentations: Dict[str, bool] = field(default_factory=lambda: {
        'flip': True, 'brightness': True, 'contrast': True
    })


# ── Action Units (FACS) ────────────────────────────────────────────────────────
# Os 12 AUs presentes no dataset DISFA+, na ordem usada como índice de classe.
AU_NAMES: List[str] = [
    'AU1',   # Inner Brow Raise
    'AU2',   # Outer Brow Raise
    'AU4',   # Brow Lowerer
    'AU5',   # Upper Lid Raiser
    'AU6',   # Cheek Raiser
    'AU9',   # Nose Wrinkler
    'AU12',  # Lip Corner Puller (sorriso)
    'AU15',  # Lip Corner Depressor
    'AU17',  # Chin Raiser
    'AU20',  # Lip Stretcher
    'AU25',  # Lips Part
    'AU26',  # Jaw Drop
]
NUM_AUS = len(AU_NAMES)  # 12
AU_INDEX: Dict[str, int] = {au: i for i, au in enumerate(AU_NAMES)}

# Descrição humana de cada AU
AU_DESCRIPTIONS: Dict[str, str] = {
    'AU1':  'Inner Brow Raise',
    'AU2':  'Outer Brow Raise',
    'AU4':  'Brow Lowerer',
    'AU5':  'Upper Lid Raiser',
    'AU6':  'Cheek Raiser',
    'AU9':  'Nose Wrinkler',
    'AU12': 'Lip Corner Puller',
    'AU15': 'Lip Corner Depressor',
    'AU17': 'Chin Raiser',
    'AU20': 'Lip Stretcher',
    'AU25': 'Lips Part',
    'AU26': 'Jaw Drop',
}

# ── FACS → Emoção ────────────────────────────────────────────────────────────
# Quais AUs (por índice) são prototípicas de cada emoção básica.
# Usado pelo motor fuzzy como entrada de mapeamento.
FACS_EMOTION_MAPPING: Dict[str, List[str]] = {
    'Happiness': ['AU6', 'AU12', 'AU25'],
    'Sadness':   ['AU1', 'AU4', 'AU15', 'AU17'],
    'Anger':     ['AU4', 'AU5', 'AU9', 'AU17'],
    'Fear':      ['AU1', 'AU2', 'AU4', 'AU5', 'AU20'],
    'Disgust':   ['AU9', 'AU15', 'AU17'],
    'Surprise':  ['AU1', 'AU2', 'AU5', 'AU26'],
}

# Intensidade FACS usada pelos rótulos DISFA/DISFA+ (0=ausente, A-E=1-5).
AU_MAX_INTENSITY: float = 5.0

# ── Configurações padrão ─────────────────────────────────────────────────────
DEFAULT_IMAGE_CONFIG = ImageConfig(
    width=224,
    height=224,
    channels=3,
    normalize=True,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

DEFAULT_MODEL_CONFIG = ModelConfig(
    model_name='YOLOv11-AU',
    input_shape=(224, 224, 3),
    num_aus=NUM_AUS,
    learning_rate=1e-4,
    epochs=50,
    batch_size=32
)

DEFAULT_TRAINING_CONFIG = TrainingConfig(
    train_split=0.8,
    validation_split=0.1,
    shuffle=True,
    accumulation_steps=4,
    use_amp=True
)


def get_dataset_path(dataset_name: str) -> Path:
    return DATA_SETS_DIR / dataset_name


def list_available_datasets() -> List[str]:
    return [d.name for d in DATA_SETS_DIR.iterdir() if d.is_dir()]
