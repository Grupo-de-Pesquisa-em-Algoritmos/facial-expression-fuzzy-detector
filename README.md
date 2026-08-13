# Face Expression Recognition With Fuzzy Logic

<div align="center">

**Reconhecimento de Expressões Faciais via Action Units (FACS) + Lógica Fuzzy**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-orange.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow.svg)]()
</div>

---

## Sobre o Projeto

Este projeto implementa um sistema de **reconhecimento de expressões faciais** em dois estágios:

1. **Detecção de Action Units (AUs)** — uma rede YOLOv11 customizada classifica quais músculos faciais estão contraídos e com qual intensidade (escala FACS 0–5), a partir do padrão FACS (*Facial Action Coding System*).
2. **Inferência de emoção via Lógica Fuzzy** — as intensidades das AUs são mapeadas para emoções básicas (Alegria, Tristeza, Raiva, Medo, Desgosto, Surpresa) por um motor fuzzy com regras linguísticas interpretáveis.

O projeto é parte do Trabalho de Conclusão de Curso (TCC) da **Universidade Estadual do Oeste do Paraná (Unioeste)** — Ciência da Computação.

---

## Arquitetura do Sistema

```
┌─────────────────────┐
│    IMAGEM ENTRADA   │
│    (3 × 224 × 224)  │
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │  YOLOv11    │
    │  Backbone   │   C3K2 + SPFF
    └──────┬──────┘
           │  P3, P4, P5
    ┌──────▼──────┐
    │  YOLOv11    │
    │  Neck (PANet│   C2PSA (atenção espacial)
    └──────┬──────┘
           │  N3, N4, N5
    ┌──────▼──────┐
    │  AU Detection   │
    │  Head       │   GAP → FC → dois ramos
    └──────┬──────┘
           │
   ┌───────┴────────┐
   │                │
┌──▼──────┐  ┌──────▼────┐
│ binary  │  │ intensity │
│ (12 AUs)│  │  (0 – 5)  │
└──┬──────┘  └──────┬────┘
   │                │
   └───────┬────────┘
           │
    ┌──────▼──────┐
    │ Motor Fuzzy │   FACS → Emoção
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   EMOÇÃO    │
    └─────────────┘
```

---

## Action Units Detectadas

O modelo detecta as 12 AUs presentes no dataset DISFA+:

| AU   | Nome FACS               | Músculo principal          |
|:-----|:------------------------|:---------------------------|
| AU1  | Inner Brow Raise        | Frontal (parte medial)     |
| AU2  | Outer Brow Raise        | Frontal (parte lateral)    |
| AU4  | Brow Lowerer            | Corrugador / Prócero       |
| AU5  | Upper Lid Raiser        | Levantador da pálpebra     |
| AU6  | Cheek Raiser            | Zigomático menor           |
| AU9  | Nose Wrinkler           | Levantador do lábio        |
| AU12 | Lip Corner Puller       | Zigomático maior (sorriso) |
| AU15 | Lip Corner Depressor    | Depressor do ângulo        |
| AU17 | Chin Raiser             | Mental                     |
| AU20 | Lip Stretcher           | Risório                    |
| AU25 | Lips Part               | Depressor do lábio inferior|
| AU26 | Jaw Drop                | Masseter (relaxamento)     |

### Mapeamento FACS → Emoção

| Emoção    | AUs prototípicas       |
|:----------|:-----------------------|
| Alegria   | AU6, AU12, AU25        |
| Tristeza  | AU1, AU4, AU15, AU17   |
| Raiva     | AU4, AU5, AU9, AU17    |
| Medo      | AU1, AU2, AU4, AU5, AU20 |
| Desgosto  | AU9, AU15, AU17        |
| Surpresa  | AU1, AU2, AU5, AU26    |

---

## Pipeline de Processamento

### Estágio 1 — Detecção de AUs (YOLOv11)

**Entrada:** imagem facial RGB (224×224)

**Head (AUDetectionHead):**
- Recebe features N3, N4, N5 do Neck
- Global Average Pooling em cada escala → concatenação
- Camada compartilhada FC → dois ramos paralelos:
  - **Ramo binário:** 12 logits → `BCEWithLogitsLoss`
  - **Ramo de intensidade:** 12 valores ∈ [0, 5] via sigmoid → `SmoothL1Loss`

**Saída:**
```python
{
    'binary_logits': Tensor(B, 12),  # sigmoid → probabilidade de ativação
    'intensity':     Tensor(B, 12),  # intensidade contínua 0–5
}
```

**Loss combinada:**

$$\mathcal{L} = \lambda_{bce} \cdot \mathcal{L}_{BCE} + \lambda_{l1} \cdot \mathcal{L}_{SmoothL1}$$

onde $\mathcal{L}_{SmoothL1}$ é calculada apenas nos frames em que a AU está ativa.

### Estágio 2 — Inferência Fuzzy

As intensidades contínuas das AUs alimentam um motor fuzzy:
- **Variáveis de entrada:** intensidade de cada AU (0–5)
- **Variáveis linguísticas:** {ausente, fraca, moderada, forte}
- **Regras:** baseadas no mapeamento FACS acima
- **Saída:** score de pertinência por emoção → classe final

---

## Dataset

O modelo é treinado no **DISFA+** (*Denver Intensity of Spontaneous Facial Actions*):

- **9 sujeitos:** SN001, SN003, SN004, SN007, SN009, SN010, SN013, SN025, SN027
- **~130 000 frames** de vídeos de expressões espontâneas
- **Anotações:** intensidade por AU por frame (0–5), feitas por especialistas FACS
- **Imagens:** faces recortadas 200×200 px

Estrutura esperada em disco:
```
datasets/archive/
├── Images/
│   └── SN001/SN001/<sessão>/<frame>.jpg
└── Labels/
    └── SN001/SN001/<sessão>/AU1.txt
                             AU2.txt
                             ...
```

---

## Instalação

### Pré-requisitos

- Python 3.11+
- CUDA 12.1+ (recomendado)

### Setup

```bash
git clone https://github.com/seu-usuario/Fer-With-Fuzzy.git
cd Fer-With-Fuzzy

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# PyTorch com CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Demais dependências
pip install -r requirements.txt
```

---

## Uso

### SDumont

Para preparar o ambiente Conda no Scratch e submeter os jobs Slurm de smoke test, treinamento e avaliação, consulte [`sdumont/README.md`](sdumont/README.md).

### Treinamento

```bash
# Padrão (6 sujeitos treino, 1 validação e 2 reservados para teste)
python main.py --mode train --epochs 50 --batch-size 4

# Com mais épocas e learning rate menor
python main.py --mode train --epochs 100 --batch-size 4 --lr 5e-5
```

### Avaliação

```bash
# Calibre os thresholds somente na validação
python main.py --mode calibrate \
  --weights checkpoints/best_model.pth \
  --thresholds-file results/thresholds.json

# Aplique-os uma única vez aos sujeitos de teste
python main.py --mode test \
  --weights checkpoints/best_model.pth \
  --thresholds-file results/thresholds.json
```

O relatório é salvo em `results/evaluation_report.txt` com precisão, recall, F1 e AP por AU, além de mAP, MAE, RMSE, correlação de Pearson e ICC(3,1).

O modelo padrão usa `base_channels=32`, adequado ao número reduzido de identidades do DISFA+. Checkpoints antigos criados com `base_channels=64` só podem ser carregados passando `--base-channels 64`; os checkpoints anteriores à correção de intensidade 0–5 devem ser retreinados.

### Demo (imagem completa)

```bash
python main.py --mode demo --image foto.jpg --weights checkpoints/best_model.pth
```

Detecta rostos automaticamente via MediaPipe e exibe as AUs ativas com suas intensidades.

### Uso programático

```python
import torch
from core import YOLOv11AUDetector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLOv11AUDetector().to(device)
checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Predição com threshold
pred = model.predict(image_tensor, binary_threshold=0.5)
# pred['binary']:    (B, 12) bool
# pred['intensity']: (B, 12) float [0, 5]
```

---

## Testes

```bash
# Testes sem necessidade do dataset DISFA+
python -m pytest tests/test_model.py tests/test_au_loss.py -v

# Suite completa (requer dataset)
python -m pytest tests/ -v
```

---

## Estrutura do Projeto

```
Fer-With-Fuzzy/
├── blocks/              # Blocos YOLOv11 (Conv, C3K2, C2PSA, SPFF, Bottleneck)
├── core/                # Backbone, Neck, AUDetectionHead, YOLOv11AUDetector
├── utils/               # DisfaDataset, AULoss, AUMetrics, Trainer, Evaluator, AUPredictor
├── config/              # Configurações globais (AUs, FACS, ImageConfig, etc.)
├── tests/               # Testes unitários
├── datasets/archive/    # Dataset DISFA+ (Images/ + Labels/)
├── checkpoints/         # Pesos do modelo
├── results/             # Relatórios de avaliação
└── main.py              # Ponto de entrada (train / calibrate / test / demo)
```

---

## Autores

- **Desenvolvedor:** Felipe Kravec Zanatta
- **Orientadora:** Adriana Postal
- **Instituição:** Unioeste — Universidade Estadual do Oeste do Paraná
- **Curso:** Ciência da Computação
- **Ano:** 2025–2026
