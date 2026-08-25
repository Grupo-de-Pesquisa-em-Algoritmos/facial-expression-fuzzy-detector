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

**Entrada:** imagem facial `grayscale` (1×224×224) nos novos experimentos, ou RGB (3×224×224) para reproduzir checkpoints antigos.

Há duas cabeças selecionáveis para uma ablação controlada com o mesmo backbone, neck, dados e loss:

- `--architecture global`: baseline monolítico com Global Average Pooling.
- `--architecture roi`: variante local-global com fusão N3/N4/N5, regiões anatômicas e `ROIAlign`.

Na variante `roi`, oito regiões fixas são definidas sobre um crop facial normalizado: sobrancelhas esquerda/direita/central, olhos+bochechas esquerda/direita, nariz, boca e queixo/mandíbula. Cada AU recebe somente suas regiões FACS relacionadas, junto de um vetor pequeno de contexto global. As regiões são prior anatômico, não bounding boxes anotadas ou preditas.

As duas cabeças produzem os mesmos ramos:

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

### Pré-processamento em grayscale

Use `--color-mode grayscale` para converter cada frame RGB em uma imagem PIL `L` de 8 bits (valores 0–255) durante a leitura. Em seguida, `ToTensor` converte o canal para `[0, 1]` e a normalização `(x - 0.5) / 0.5` o leva a `[-1, 1]`. O primeiro `Conv2d` recebe realmente um canal; a imagem não é replicada três vezes e o dataset original não é regravado.

Isso remove matiz e saturação como atalhos, mas não torna o modelo automaticamente independente de tom de pele: luminância, exposição e contraste ainda podem se correlacionar com grupos demográficos. O experimento deve comparar RGB e grayscale com o mesmo split e relatar as métricas também por grupos de tom de pele/exposição quando essas anotações estiverem disponíveis.

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
- **Imagens neste repositório:** frames completos; um cache reprodutível de caixas faciais gera os crops usados pelo ROIAlign

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
# Gere uma vez as caixas dos rostos (salva datasets/archive/face_boxes.json)
python tools/precompute_face_boxes.py --data-dir datasets/archive

# ROIAlign (6 sujeitos treino, 1 validação e 2 reservados para teste)
python main.py --mode train --architecture roi --epochs 50 --batch-size 32 \
  --color-mode grayscale \
  --save-dir checkpoints/roi-01

# Baseline global com os mesmos crops para comparação justa entre cabeças
python main.py --mode train --architecture global --face-crops \
  --color-mode grayscale \
  --epochs 50 --batch-size 32 \
  --save-dir checkpoints/global-01
```

`--architecture roi` habilita os crops automaticamente. `--face-boxes` aceita um cache em outro caminho e `--face-margin` controla a margem ao redor do rosto (padrão `0.20`). O modo global mantém, por compatibilidade, os frames completos quando `--face-crops` não é informado; use a flag no experimento comparativo.

### Avaliação

```bash
# Calibre os thresholds somente na validação
python main.py --mode calibrate \
  --architecture roi \
  --color-mode grayscale \
  --weights checkpoints/roi-01/best_model.pth \
  --thresholds-file results/thresholds-roi-01.json

# Aplique-os uma única vez aos sujeitos de teste
python main.py --mode test \
  --architecture roi \
  --color-mode grayscale \
  --weights checkpoints/roi-01/best_model.pth \
  --thresholds-file results/thresholds-roi-01.json
```

O relatório é salvo em `results/evaluation_report.txt` com precisão, recall, F1 e AP por AU, além de mAP, MAE, RMSE, correlação de Pearson e ICC(3,1).

O modelo padrão usa `base_channels=32`. A variante ROI usa ainda `--roi-channels 128 --roi-size 3`. Esses valores, a configuração de crop e o contrato de cor/normalização são persistidos no checkpoint e validados ao carregar os pesos. Um checkpoint treinado em RGB não pode ser carregado por um modelo grayscale (nem o inverso); para reproduzir pesos antigos, informe `--color-mode rgb` em todas as etapas.

### Demo (imagem completa)

```bash
python main.py --mode demo --architecture roi --color-mode grayscale --image foto.jpg \
  --weights checkpoints/roi-01/best_model.pth \
  --thresholds-file results/thresholds-roi-01.json
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
