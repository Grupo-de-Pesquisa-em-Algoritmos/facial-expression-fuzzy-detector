#!/bin/bash

# Configuração compartilhada pelos jobs do SDumont.
# Este arquivo é carregado depois que o Slurm já alocou os recursos.

set -euo pipefail

if [[ -z "${SCRATCH:-}" ]]; then
    echo "ERRO: a variável SCRATCH não está definida." >&2
    exit 1
fi

PROJECT_DIR="${SDUMONT_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_DIR="${SDUMONT_CONDA_ENV:-$SCRATCH/conda-envs/fer-with-fuzzy}"
DATA_DIR="${SDUMONT_DATA_DIR:-$PROJECT_DIR/datasets/archive}"
ANACONDA_MODULE="${SDUMONT_ANACONDA_MODULE:-anaconda3/2024.02_sequana}"

module purge
module load "$ANACONDA_MODULE"
source activate "$ENV_DIR"

if [[ ! -f "$PROJECT_DIR/main.py" ]]; then
    echo "ERRO: main.py não encontrado em $PROJECT_DIR" >&2
    echo "Submeta o job a partir da raiz do projeto ou defina SDUMONT_PROJECT_DIR." >&2
    exit 1
fi

if [[ ! -d "$DATA_DIR/Images" || ! -d "$DATA_DIR/Labels" ]]; then
    echo "ERRO: dataset DISFA+ inválido em $DATA_DIR" >&2
    echo "Esperado: $DATA_DIR/Images e $DATA_DIR/Labels" >&2
    exit 1
fi

cd "$PROJECT_DIR"
mkdir -p checkpoints results sdumont/logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

echo "Job:       ${SLURM_JOB_ID:-sem-id}"
echo "Nó(s):     ${SLURM_JOB_NODELIST:-não informado}"
echo "Projeto:   $PROJECT_DIR"
echo "Dataset:   $DATA_DIR"
echo "Ambiente:  $ENV_DIR"
echo "Python:    $(command -v python)"

nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'indisponível'); assert torch.cuda.is_available(), 'PyTorch não detectou a GPU alocada'"
