#!/bin/bash

# Execute uma vez em um nó de login, com o projeto já copiado para o SCRATCH.

set -euo pipefail

if [[ -z "${SCRATCH:-}" ]]; then
    echo "ERRO: a variável SCRATCH não está definida." >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="${SDUMONT_CONDA_ENV:-$SCRATCH/conda-envs/fer-with-fuzzy}"
ANACONDA_MODULE="${SDUMONT_ANACONDA_MODULE:-anaconda3/2024.02_sequana}"

module purge
module load "$ANACONDA_MODULE"

if [[ ! -d "$ENV_DIR" ]]; then
    mkdir -p "$(dirname -- "$ENV_DIR")"
    conda create --yes --prefix "$ENV_DIR" python=3.11
else
    echo "Ambiente existente será atualizado: $ENV_DIR"
fi

source activate "$ENV_DIR"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$PROJECT_DIR/requirements-sdumont.txt"

mkdir -p "$PROJECT_DIR/checkpoints" "$PROJECT_DIR/results" "$PROJECT_DIR/sdumont/logs"

python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA runtime do wheel:', torch.version.cuda)"

echo
echo "Ambiente pronto em: $ENV_DIR"
echo "Não deixe o Conda ativado ao chamar sbatch; os jobs ativam o ambiente internamente."
