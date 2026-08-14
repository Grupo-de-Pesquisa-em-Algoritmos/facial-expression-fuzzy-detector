"""
main.py — YOLOv11 para Detecção de Action Units (FACS)

Uso:
    python main.py --mode train --epochs 50 --batch-size 4
    python main.py --mode test  --weights checkpoints/best_model.pth
    python main.py --mode demo  --image caminho/para/foto.jpg
"""

import argparse
import json
import sys
import torch
from pathlib import Path

from config import AU_NAMES, DEFAULT_IMAGE_CONFIG
from core import YOLOv11AUDetector
from utils import AULoss, Trainer, Evaluator, AUPredictor
from utils.dataset_loader import (
    DEFAULT_TEST_SUBJECTS,
    DEFAULT_TRAIN_SUBJECTS,
    DEFAULT_VAL_SUBJECTS,
    create_dataloaders,
    create_eval_dataloader,
)


def configure_console_encoding():
    """Evita UnicodeEncodeError nos consoles Windows que ainda usam cp1252."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')


def select_device(require_cuda=False):
    """Seleciona CUDA quando disponível e falha cedo em jobs que exigem GPU."""
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não está disponível. Verifique a alocação de GPU, os módulos "
            "carregados e a instalação do PyTorch."
        )
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resolve_face_crops(args) -> bool:
    """ROI usa crops por padrão; a flag permite uma ablação global justa."""
    return args.face_crops if args.face_crops is not None else args.architecture == 'roi'


def face_boxes_path(args) -> Path | None:
    return Path(args.face_boxes) if args.face_boxes else None


# ─── Construtores ─────────────────────────────────────────────────────────────

def setup_model_and_criterion(args, device):
    """Instancia o modelo AU e a função de perda."""
    model = YOLOv11AUDetector(
        in_channels=3,
        base_channels=args.base_channels,
        architecture=args.architecture,
        roi_channels=args.roi_channels,
        roi_size=args.roi_size,
    ).to(device)
    model.preprocessing_config = {
        'face_crops': resolve_face_crops(args),
        'face_margin': args.face_margin,
    }

    # pos_weight calculado a partir do dataset de treino
    # (None → BCE sem pesos; será sobrescrito durante setup_dataloaders quando possível)
    criterion = AULoss(
        pos_weight=None,
        lambda_binary=1.0,
        lambda_intensity=0.5,
    )

    return model, criterion


def setup_dataloaders(args):
    """Cria DataLoaders de treino e validação a partir do DISFA+."""
    print("\n📂 Carregando dataset DISFA+...")
    train_loader, val_loader = create_dataloaders(
        subjects_train=args.train_subjects,
        subjects_val=args.val_subjects,
        img_config=DEFAULT_IMAGE_CONFIG,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        disfa_dir=Path(args.data_dir),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        crop_faces=resolve_face_crops(args),
        face_boxes_file=face_boxes_path(args),
        face_margin=args.face_margin,
    )
    print(f"   Treino : {len(train_loader.dataset):>6} amostras")
    print(f"   Val    : {len(val_loader.dataset):>6} amostras")
    print(f"   Sujeitos treino: {args.train_subjects}")
    print(f"   Sujeitos val:    {args.val_subjects}")
    print(f"   Crop facial:     {resolve_face_crops(args)}")
    return train_loader, val_loader


def load_thresholds(path: str | Path):
    """Carrega thresholds por AU do JSON produzido no modo calibrate."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    values = payload.get('thresholds', payload)
    try:
        return [float(values[au]) for au in AU_NAMES]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Arquivo de thresholds inválido: {path}") from exc


def resolve_thresholds(args):
    threshold_path = Path(args.thresholds_file) if args.thresholds_file else None
    if threshold_path and threshold_path.is_file():
        print(f"🎚️  Thresholds calibrados: {threshold_path}")
        return load_thresholds(threshold_path)
    if threshold_path and args.mode == 'test':
        raise FileNotFoundError(f"Arquivo de thresholds não encontrado: {threshold_path}")
    return args.conf_threshold


def load_checkpoint_into_model(model, checkpoint_path, device):
    """Carrega pesos e rejeita cedo uma arquitetura incompatível."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_config = checkpoint.get('model_config', {})
    current_config = model.export_config()
    for key in ('architecture', 'base_channels', 'roi_channels', 'roi_size'):
        if key in saved_config and saved_config[key] != current_config[key]:
            raise ValueError(
                f"Checkpoint usa {key}={saved_config[key]!r}, mas o modelo foi "
                f"criado com {key}={current_config[key]!r}. Ajuste o argumento correspondente."
            )
    saved_preprocessing = checkpoint.get('preprocessing_config', {})
    current_preprocessing = getattr(model, 'preprocessing_config', {})
    for key in ('face_crops', 'face_margin'):
        if key in saved_preprocessing and saved_preprocessing[key] != current_preprocessing.get(key):
            raise ValueError(
                f"Checkpoint usa {key}={saved_preprocessing[key]!r}, mas a execução "
                f"usa {key}={current_preprocessing.get(key)!r}. Ajuste o pré-processamento."
            )
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state)
    return checkpoint


# ─── Modos ────────────────────────────────────────────────────────────────────

def train_model(args):
    """Loop completo de treinamento."""
    device = select_device(args.require_cuda)
    print(f"🖥️  Device: {device}")
    print(f"🧠 Arquitetura: {args.architecture}")

    train_loader, val_loader = setup_dataloaders(args)
    model, criterion = setup_model_and_criterion(args, device)

    # Atualizar pos_weight com base no dataset de treino
    pos_weight = train_loader.dataset.compute_pos_weight(
        device=str(device),
        mode=args.pos_weight_mode,
        max_weight=args.max_pos_weight,
    )
    print(f"⚖️  pos_weight ({args.pos_weight_mode}): {pos_weight.detach().cpu().tolist()}")
    criterion.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')

    # Retomar treinamento se solicitado
    if args.resume:
        print(f"📥 Retomando checkpoint: {args.resume}")
        checkpoint = load_checkpoint_into_model(model, args.resume, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    effective_batch = 32
    accumulation_steps = max(1, effective_batch // args.batch_size)
    print(f"\n⚙️  Gradient Accumulation: {accumulation_steps}x "
          f"(effective batch = {args.batch_size * accumulation_steps})")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        accumulation_steps=accumulation_steps,
        use_amp=True,
        early_stopping_patience=args.early_stopping_patience,
        monitor=args.monitor,
    )
    trainer.train()


def test_model(args):
    """Avalia o modelo num set de teste e grava relatório."""
    device = select_device(args.require_cuda)

    print("=" * 70)
    print("Avaliação — YOLOv11 Action Units")
    print("=" * 70)

    model, _ = setup_model_and_criterion(args, device)

    print(f"\n📥 Carregando pesos: {args.weights}")
    checkpoint = load_checkpoint_into_model(model, args.weights, device)

    test_loader = create_eval_dataloader(
        subjects=args.test_subjects,
        img_config=DEFAULT_IMAGE_CONFIG,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        disfa_dir=Path(args.data_dir),
        max_samples=args.max_test_samples,
        crop_faces=resolve_face_crops(args),
        face_boxes_file=face_boxes_path(args),
        face_margin=args.face_margin,
    )

    print(f"   Sujeitos teste: {args.test_subjects}")
    evaluator = Evaluator(model, device=str(device), results_dir=args.results_dir)
    results = evaluator.evaluate_and_save(
        test_loader,
        thresholds=resolve_thresholds(args),
    )

    from utils.metrics import AUMetrics
    print("\n" + AUMetrics.format_summary(results))


def calibrate_model(args):
    """Calibra um threshold por AU usando somente os sujeitos de validação."""
    device = select_device(args.require_cuda)
    model, _ = setup_model_and_criterion(args, device)
    checkpoint = load_checkpoint_into_model(model, args.weights, device)

    val_loader = create_eval_dataloader(
        subjects=args.val_subjects,
        img_config=DEFAULT_IMAGE_CONFIG,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        disfa_dir=Path(args.data_dir),
        max_samples=args.max_val_samples,
        crop_faces=resolve_face_crops(args),
        face_boxes_file=face_boxes_path(args),
        face_margin=args.face_margin,
    )
    threshold_path = Path(args.thresholds_file or 'results/thresholds.json')
    evaluator = Evaluator(model, device=str(device), results_dir=threshold_path.parent)
    thresholds, results = evaluator.calibrate_and_save(
        val_loader,
        filename=threshold_path.name,
        subjects=args.val_subjects,
    )
    from utils.metrics import AUMetrics
    print("\n" + AUMetrics.format_summary(results))
    print(f"\nThresholds: {dict(zip(AU_NAMES, thresholds.tolist()))}")


def demo(args):
    """Demonstração AU em imagem completa (com detecção de rosto via MediaPipe)."""
    device = select_device(args.require_cuda)

    print("=" * 70)
    print("Demo — Detecção de Action Units")
    print("=" * 70)

    model, _ = setup_model_and_criterion(args, device)

    print(f"\n📥 Carregando pesos: {args.weights}")
    checkpoint = load_checkpoint_into_model(model, args.weights, device)

    predictor = AUPredictor(
        model,
        device=str(device),
        threshold=resolve_thresholds(args),
        img_config=DEFAULT_IMAGE_CONFIG,
    )

    print(f"\n🔍 Analisando: {args.image}")
    face_results = predictor.predict_image(args.image)

    from utils.inference import format_au_results
    report = format_au_results(face_results)
    print("\n" + report)

    if args.output:
        Path(args.output).write_text(report, encoding='utf-8')
        print(f"\n💾 Resultado salvo em: {args.output}")

    print("=" * 70)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    configure_console_encoding()
    parser = argparse.ArgumentParser(
        description='YOLOv11 — Detecção de Action Units (FACS / DISFA+)'
    )

    parser.add_argument('--mode', default='train', choices=['train', 'calibrate', 'test', 'demo'])

    # Modelo
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--architecture', choices=['global', 'roi'], default='global')
    parser.add_argument('--roi-channels', type=int, default=128)
    parser.add_argument('--roi-size', type=int, default=3)
    parser.add_argument('--weights', default='checkpoints/best_model.pth')
    parser.add_argument('--resume', default='')

    # Treinamento
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--save-dir', default='checkpoints')
    parser.add_argument('--data-dir', default='datasets/archive')
    parser.add_argument('--face-boxes', default='')
    parser.add_argument('--face-margin', type=float, default=0.20)
    parser.add_argument(
        '--face-crops',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='usa crops faciais; por padrão é ligado para roi e desligado para global',
    )
    parser.add_argument('--max-train-samples', type=int)
    parser.add_argument('--max-val-samples', type=int)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--train-subjects', nargs='+', default=DEFAULT_TRAIN_SUBJECTS)
    parser.add_argument('--val-subjects', nargs='+', default=DEFAULT_VAL_SUBJECTS)
    parser.add_argument('--test-subjects', nargs='+', default=DEFAULT_TEST_SUBJECTS)
    parser.add_argument('--max-test-samples', type=int)
    parser.add_argument('--pos-weight-mode', choices=['none', 'sqrt', 'balanced'], default='sqrt')
    parser.add_argument('--max-pos-weight', type=float, default=5.0)
    parser.add_argument('--early-stopping-patience', type=int, default=7)
    parser.add_argument('--monitor', choices=['map', 'f1_macro', 'val_loss'], default='map')
    parser.add_argument('--results-dir', default='results')

    # Demo
    parser.add_argument('--image', default='')
    parser.add_argument('--output', default='')
    parser.add_argument('--conf-threshold', type=float, default=0.5)
    parser.add_argument('--thresholds-file', default='')

    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'calibrate':
        calibrate_model(args)
    elif args.mode == 'test':
        test_model(args)
    elif args.mode == 'demo':
        if not args.image:
            parser.error('--image é obrigatório no modo demo')
        demo(args)


if __name__ == '__main__':
    main()
