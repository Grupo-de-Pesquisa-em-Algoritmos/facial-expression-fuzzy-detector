# Execução no SDumont

Este diretório contém os jobs Slurm do projeto:

| Arquivo | Fila padrão | Uso |
|---|---|---|
| `smoke.srm` | `sequana_gpu_dev` | Teste curto com 256 amostras de treino e 64 de validação |
| `train.srm` | `sequana_gpu` | Treino completo, 50 épocas por padrão |
| `evaluate.srm` | `sequana_gpu` | Avaliação de um checkpoint |

Os três jobs usam um nó, uma GPU V100 e um processo Python. O código atual não implementa DDP, portanto solicitar mais GPUs ou mais nós não acelera o treinamento e desperdiça UAs.

## 1. Conferir a conta e as filas

Depois de conectar à VPN e entrar por SSH, execute no nó de login:

```bash
echo "$SCRATCH"
sacctmgr list account "$GROUPNAME" format=account,descr -P
sacctmgr list user "$USER" -s format=partition%20,MaxJobs,MaxSubmit,MaxNodes,MaxCPUs,MaxWall
module avail anaconda3
```

Os nomes e limites das filas dependem do programa de alocação do projeto. Se `sequana_gpu` ou `sequana_gpu_dev` não aparecerem, ajuste `--partition` na submissão ou solicite acesso ao suporte do SDumont.

## 2. Manter código, ambiente e dados no Scratch

Os nós computacionais não acessam o Home. Clone ou copie o projeto para o Scratch e faça os comandos seguintes a partir da raiz do projeto:

```bash
cd "$SCRATCH"
git clone URL_DO_REPOSITORIO Fer-With-Fuzzy
cd Fer-With-Fuzzy
```

Coloque o DISFA+ em um diretório do Scratch com esta estrutura:

```text
$SCRATCH/datasets/disfa-plus/
├── Images/
└── Labels/
```

O caminho também pode continuar sendo `datasets/archive` dentro do projeto. Para manter o dataset separado do Git, passe `SDUMONT_DATA_DIR` ao submeter os jobs, como mostrado abaixo.

## 3. Criar o ambiente

Execute uma vez no nó de login, que possui acesso à Internet:

```bash
bash sdumont/setup_environment.sh
```

Por padrão, o ambiente será criado em `$SCRATCH/conda-envs/fer-with-fuzzy`, com Python 3.11 e os wheels PyTorch/CUDA 12.6 definidos em `requirements-sdumont.txt`. Não é necessário carregar um módulo CUDA ou cuDNN adicional: o wheel do PyTorch contém o runtime, enquanto o driver é fornecido pelo nó GPU.

Se o módulo padrão não existir na sua conta, escolha um listado por `module avail anaconda3`:

```bash
SDUMONT_ANACONDA_MODULE=anaconda3/VERSAO bash sdumont/setup_environment.sh
```

O ambiente não deve estar ativado no shell que chama `sbatch`; cada job carrega e ativa o próprio ambiente.

## 4. Validar e executar

Sempre submeta a partir da raiz do repositório, pois os scripts usam `$SLURM_SUBMIT_DIR` como diretório do projeto.

Primeiro valide a estimativa do escalonador e execute o smoke test na fila de desenvolvimento:

```bash
sbatch --test-only sdumont/smoke.srm
sbatch --export=ALL,SDUMONT_DATA_DIR="$SCRATCH/datasets/disfa-plus" sdumont/smoke.srm
```

Depois do smoke test terminar com estado `COMPLETED`, submeta o treino completo:

```bash
sbatch --test-only sdumont/train.srm
sbatch --export=ALL,SDUMONT_DATA_DIR="$SCRATCH/datasets/disfa-plus" sdumont/train.srm
```

Para avaliar o melhor checkpoint:

```bash
sbatch --export=ALL,SDUMONT_DATA_DIR="$SCRATCH/datasets/disfa-plus",SDUMONT_WEIGHTS="$SCRATCH/Fer-With-Fuzzy/checkpoints/best_model.pth" sdumont/evaluate.srm
```

## 5. Acompanhar

```bash
squeue -u "$USER"
squeue --start -j JOB_ID
scontrol show jobid JOB_ID -dd
sacct -j JOB_ID --format=JobID,JobName,Partition,State,Elapsed,ExitCode,AllocTRES%50
tail -f sdumont/logs/fer-smoke-JOB_ID.out
```

Para cancelar um job:

```bash
scancel JOB_ID
```

Os logs ficam em `sdumont/logs/`, checkpoints em `checkpoints/` e relatórios em `results/`.

## 6. Ajustes sem editar os scripts

Os parâmetros mais comuns podem ser enviados como variáveis do job:

```bash
sbatch --export=ALL,SDUMONT_DATA_DIR="$SCRATCH/datasets/disfa-plus",SDUMONT_EPOCHS=100,SDUMONT_BATCH_SIZE=8,SDUMONT_LR=5e-5 sdumont/train.srm
```

Para continuar a partir dos pesos de um checkpoint:

```bash
sbatch --export=ALL,SDUMONT_DATA_DIR="$SCRATCH/datasets/disfa-plus",SDUMONT_RESUME="$SCRATCH/Fer-With-Fuzzy/checkpoints/best_model.pth" sdumont/train.srm
```

Variáveis aceitas:

| Variável | Padrão no treino | Efeito |
|---|---:|---|
| `SDUMONT_DATA_DIR` | `datasets/archive` | Raiz do DISFA+ |
| `SDUMONT_CONDA_ENV` | `$SCRATCH/conda-envs/fer-with-fuzzy` | Ambiente Conda |
| `SDUMONT_EPOCHS` | `50` | Número de épocas |
| `SDUMONT_BATCH_SIZE` | `4` | Batch por iteração |
| `SDUMONT_NUM_WORKERS` | `8` | Processos de leitura do dataset |
| `SDUMONT_LR` | `1e-4` | Learning rate |
| `SDUMONT_SAVE_DIR` | `checkpoints` | Diretório de checkpoints |
| `SDUMONT_RESUME` | vazio | Checkpoint inicial do treino |
| `SDUMONT_WEIGHTS` | `checkpoints/best_model.pth` | Pesos usados na avaliação |

Opções Slurm do arquivo podem ser substituídas na linha de comando. Exemplo para uma fila permitida pela sua conta e um limite de 12 horas:

```bash
sbatch --partition=MINHA_FILA_GPU --time=12:00:00 sdumont/train.srm
```

Defina o menor wall-clock realista: isso favorece o backfill, mas o job é cancelado se ultrapassar o limite. A fila `sequana_gpu_dev` permite apenas 20 minutos e deve ser usada para testes, não para o treino completo.

## Referências oficiais

- [Características e armazenamento](https://github.com/lncc-sered/manual-sdumont/wiki/02-%E2%80%90-Caracter%C3%ADsticas)
- [Acesso e submissão](https://github.com/lncc-sered/manual-sdumont/wiki/03-%E2%80%90-Informa%C3%A7%C3%B5es-sobre-acesso-e-submiss%C3%A3o)
- [Módulos de ambiente](https://github.com/lncc-sered/manual-sdumont/wiki/04-%E2%80%90-M%C3%B3dulos-de-Ambiente)
- [Filas e política de uso](https://github.com/lncc-sered/manual-sdumont/wiki/06-%E2%80%90--Gerenciador-de-filas)
- [Submissão de jobs e GPUs](https://github.com/lncc-sered/manual-sdumont/wiki/07-%E2%80%90-Submeter-Jobs)
- [Status e utilização](https://github.com/lncc-sered/manual-sdumont/wiki/08-%E2%80%90-Verificando-status-e-utiliza%C3%A7%C3%A3o)
- [Ambientes virtuais Python](https://github.com/lncc-sered/manual-sdumont/wiki/09-%E2%80%90-FAQ#pacotes-pythonr-e-ambientes-virtuais)
