"""
main.py — YOLOv11 para Detecção de Action Units (FACS)

Uso:
    python main.py --mode train --epochs 50 --batch-size 4
    python main.py --mode test  --weights checkpoints/best_model.pth
    python main.py --mode demo  --image caminho/para/foto.jpg
"""

import argparse
import torch
from pathlib import Path

from config import DEFAULT_IMAGE_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG
from core import YOLOv11AUDetector
from utils import DisfaDataset, AULoss, Trainer, Evaluator, AUPredictor
from utils.dataset_loader import create_dataloaders, DISFA_SUBJECTS


def select_device(require_cuda=False):
    """Seleciona CUDA quando disponível e falha cedo em jobs que exigem GPU."""
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não está disponível. Verifique a alocação de GPU, os módulos "
            "carregados e a instalação do PyTorch."
        )
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── Construtores ─────────────────────────────────────────────────────────────

def setup_model_and_criterion(args, device):
    """Instancia o modelo AU e a função de perda."""
    model = YOLOv11AUDetector(
        in_channels=3,
        base_channels=args.base_channels,
    ).to(device)

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
        img_config=DEFAULT_IMAGE_CONFIG,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        disfa_dir=Path(args.data_dir),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    print(f"   Treino : {len(train_loader.dataset):>6} amostras")
    print(f"   Val    : {len(val_loader.dataset):>6} amostras")
    return train_loader, val_loader


# ─── Modos ────────────────────────────────────────────────────────────────────

def train_model(args):
    """Loop completo de treinamento."""
    device = select_device(args.require_cuda)
    print(f"🖥️  Device: {device}")

    train_loader, val_loader = setup_dataloaders(args)
    model, criterion = setup_model_and_criterion(args, device)

    # Atualizar pos_weight com base no dataset de treino
    pos_weight = train_loader.dataset.compute_pos_weight(device=str(device))
    criterion.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')

    # Retomar treinamento se solicitado
    if args.resume:
        print(f"📥 Retomando checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

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
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Usar o último sujeito como set de teste (separado do treino)
    from utils.dataset_loader import create_dataloaders
    _, test_loader = create_dataloaders(
        subjects_val=[DISFA_SUBJECTS[-1]],
        img_config=DEFAULT_IMAGE_CONFIG,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        disfa_dir=Path(args.data_dir),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    evaluator = Evaluator(model, device=str(device), results_dir='results')
    results = evaluator.evaluate_and_save(test_loader)

    from utils.metrics import AUMetrics
    print("\n" + AUMetrics.format_summary(results))


def demo(args):
    """Demonstração AU em imagem completa (com detecção de rosto via MediaPipe)."""
    device = select_device(args.require_cuda)

    print("=" * 70)
    print("Demo — Detecção de Action Units")
    print("=" * 70)

    model, _ = setup_model_and_criterion(args, device)

    print(f"\n📥 Carregando pesos: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    predictor = AUPredictor(
        model,
        device=str(device),
        threshold=args.conf_threshold,
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
    parser = argparse.ArgumentParser(
        description='YOLOv11 — Detecção de Action Units (FACS / DISFA+)'
    )

    parser.add_argument('--mode', default='train', choices=['train', 'test', 'demo'])

    # Modelo
    parser.add_argument('--base-channels', type=int, default=64)
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
    parser.add_argument('--max-train-samples', type=int)
    parser.add_argument('--max-val-samples', type=int)
    parser.add_argument('--require-cuda', action='store_true')

    # Demo
    parser.add_argument('--image', default='')
    parser.add_argument('--output', default='')
    parser.add_argument('--conf-threshold', type=float, default=0.5)

    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)
    elif args.mode == 'demo':
        if not args.image:
            parser.error('--image é obrigatório no modo demo')
        demo(args)


if __name__ == '__main__':
    main()
