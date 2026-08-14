"""
Trainer - Gerenciador de Treinamento do Modelo

Responsável por:
- Training loop
- Validação
- Checkpoints
- Logging
"""

import torch
import torch.nn as nn
from pathlib import Path
import time
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from utils.metrics import AUMetrics


class Trainer:
    """
    Classe para gerenciar o treinamento do modelo
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs,
        save_dir='checkpoints',
        accumulation_steps=4,
        use_amp=True,
        early_stopping_patience=7,
        monitor='map',
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp and torch.cuda.is_available()
        # ✅ CORRIGIDO: GradScaler atualizado para PyTorch 2.9+
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        if monitor not in {'map', 'f1_macro', 'val_loss'}:
            raise ValueError("monitor deve ser 'map', 'f1_macro' ou 'val_loss'")
        self.monitor = monitor
        self.early_stopping_patience = max(0, early_stopping_patience)
        self.best_score = float('inf') if monitor == 'val_loss' else float('-inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_binary_loss': [],
            'train_intensity_loss': [],
            'val_binary_loss': [],
            'val_intensity_loss': [],
            'val_f1_macro': [],
            'val_map': [],
            'val_mae': [],
        }

    def train_epoch(self, epoch):
        """Treina uma época"""
        self.model.train()
        total_loss = 0
        total_binary = 0
        total_intensity = 0

        train_pbar = tqdm(self.train_loader, desc=f"Época {epoch+1}/{self.num_epochs} [Train]", leave=True)

        for batch_idx, batch in enumerate(train_pbar):
            images    = batch['image'].to(self.device)
            binary    = batch['binary'].to(self.device)
            intensity = batch['intensity'].to(self.device)

            with autocast('cuda', enabled=self.use_amp):
                predictions = self.model(images)
                targets = {'binary': binary, 'intensity': intensity}
                losses = self.criterion(predictions, targets)
                loss = losses['loss'] / self.accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Atualizar pesos a cada N steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

            total_loss     += losses['loss'].item()
            total_binary   += losses['binary_loss'].item()
            total_intensity += losses['intensity_loss'].item()

            train_pbar.set_postfix({
                'loss':  f'{losses["loss"].item():.3f}',
                'bce':   f'{losses["binary_loss"].item():.3f}',
                'l1':    f'{losses["intensity_loss"].item():.3f}',
            })
        
        remaining_batches = len(self.train_loader) % self.accumulation_steps
        if remaining_batches != 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        self.optimizer.zero_grad()

        num_batches = len(self.train_loader)
        return total_loss / num_batches, total_binary / num_batches, total_intensity / num_batches

    @torch.no_grad()
    def validate(self, epoch):
        """Valida o modelo"""
        self.model.eval()
        total_loss = 0
        total_binary = 0
        total_intensity = 0
        metrics = AUMetrics()

        pbar = tqdm(self.val_loader, desc=f'Época {epoch+1}/{self.num_epochs} [Val]')

        for batch in pbar:
            images    = batch['image'].to(self.device)
            binary    = batch['binary'].to(self.device)
            intensity = batch['intensity'].to(self.device)

            predictions = self.model(images)
            targets = {'binary': binary, 'intensity': intensity}
            losses = self.criterion(predictions, targets)
            metrics.update(predictions, targets)

            total_loss      += losses['loss'].item()
            total_binary    += losses['binary_loss'].item()
            total_intensity += losses['intensity_loss'].item()

            pbar.set_postfix({
                'loss': f'{losses["loss"].item():.3f}',
                'bce':  f'{losses["binary_loss"].item():.3f}',
                'l1':   f'{losses["intensity_loss"].item():.3f}',
            })

        n = len(self.val_loader)
        result = metrics.compute()
        result.update({
            'val_loss': total_loss / n,
            'binary_loss': total_binary / n,
            'intensity_loss': total_intensity / n,
        })
        return result

    def save_checkpoint(self, epoch, val_metrics, is_best=False):
        """Salva checkpoint do modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_metrics['val_loss'],
            'val_metrics': {
                key: value for key, value in val_metrics.items()
                if isinstance(value, (int, float))
            },
            'monitor': self.monitor,
            'best_score': self.best_score,
            'history': self.history
        }
        if hasattr(self.model, 'export_config'):
            checkpoint['model_config'] = self.model.export_config()
        if hasattr(self.model, 'preprocessing_config'):
            checkpoint['preprocessing_config'] = self.model.preprocessing_config
        
        # Salvar checkpoint regular
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Salvar melhor modelo
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f'✅ Melhor modelo salvo com {self.monitor}={val_metrics[self.monitor]:.4f}')

    def _is_better(self, value):
        return value < self.best_score if self.monitor == 'val_loss' else value > self.best_score

    def train(self):
        """Loop de treinamento completo"""
        print("=" * 70)
        print("Iniciando Treinamento do YOLOv11 para Action Units")
        print("=" * 70)
        print(f"📊 Dataset:")
        print(f"   Treino: {len(self.train_loader.dataset)} amostras")
        print(f"   Validação: {len(self.val_loader.dataset)} amostras")
        print(f"\n⚙️  Configurações:")
        print(f"   Épocas: {self.num_epochs}")
        print(f"   Batch size: {self.train_loader.batch_size}")
        print(f"   Accumulation steps: {self.accumulation_steps}")
        print(f"   Effective batch size: {self.train_loader.batch_size * self.accumulation_steps}")
        print(f"   Mixed Precision (AMP): {'Ativado' if self.use_amp else 'Desativado'}")
        print(f"   Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"   Device: {self.device}")
        print("=" * 70 + "\n")
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # Treinar
            train_loss, train_bce, train_l1 = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_binary_loss'].append(train_bce)
            self.history['train_intensity_loss'].append(train_l1)

            # Validar
            val_metrics = self.validate(epoch)
            val_loss = val_metrics['val_loss']
            val_bce = val_metrics['binary_loss']
            val_l1 = val_metrics['intensity_loss']
            self.history['val_loss'].append(val_loss)
            self.history['val_binary_loss'].append(val_bce)
            self.history['val_intensity_loss'].append(val_l1)
            self.history['val_f1_macro'].append(val_metrics['f1_macro'])
            self.history['val_map'].append(val_metrics['map'])
            self.history['val_mae'].append(val_metrics['mae_mean'])

            # Atualizar learning rate
            self.scheduler.step(val_loss)

            epoch_time = time.time() - start_time

            # Log
            print(f"\n📈 Época {epoch+1}/{self.num_epochs} - {epoch_time:.2f}s")
            print(f"   Train Loss: {train_loss:.4f} (bce: {train_bce:.4f}, l1: {train_l1:.4f})")
            print(f"   Val Loss:   {val_loss:.4f} (bce: {val_bce:.4f}, l1: {val_l1:.4f})")
            print(f"   Val Macro F1: {val_metrics['f1_macro']:.4f} | "
                  f"mAP: {val_metrics['map']:.4f} | MAE: {val_metrics['mae_mean']:.4f}")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Salvar checkpoint
            monitor_value = val_metrics[self.monitor]
            is_best = self._is_better(monitor_value)
            if is_best:
                self.best_score = monitor_value
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            print()

            if (
                self.early_stopping_patience > 0
                and self.epochs_without_improvement >= self.early_stopping_patience
            ):
                print(
                    f"⏹ Early stopping: {self.monitor} não melhorou por "
                    f"{self.early_stopping_patience} épocas."
                )
                break
        
        print("=" * 70)
        print(f"✅ Treinamento concluído!")
        print(f"   Melhor {self.monitor}: {self.best_score:.4f}")
        print(f"   Checkpoints salvos em: {self.save_dir}")
        print("=" * 70)
