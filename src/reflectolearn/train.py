from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(self,
        model: nn.Module, train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str | Path,
        lr: float = 2e-4,  # Fourier-feature models are more stable with a slightly lower LR
        weight_decay: float = 1e-4,
        patience: int = 20,
        verbose: bool = True,   # set False to silence per-trial logs (e.g. during HPO)
    ):
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.checkpoint_dir = Path(save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # AdamW: decoupled weight decay makes the L2 regularization behave correctly.
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.criterion = nn.MSELoss()

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', patience=7, factor=0.5
        )

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = patience

        self.history = {"train": [], "val": [], "lr": []}

        # Mixed-precision training (disabled).
        self.scaler = None
        if self.verbose:
            print(f"🚀 Trainer initialized on {self.device}")
            print(f"   - Augmentation: {'Enabled' if getattr(train_loader.dataset, 'augment', False) else 'Disabled'}")

    def _train_epoch(self, epoch: int, total_epochs: int) -> float:
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)

        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler:
                with torch.amp.autocast('cuda'):
                    preds = self.model(inputs)
                    loss = self.criterion(preds, targets)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(inputs)
                loss = self.criterion(preds, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        running_loss = 0.0
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            preds = self.model(inputs)
            loss = self.criterion(preds, targets)
            running_loss += loss.item()
        return running_loss / len(self.val_loader)

    def train(
        self,
        epochs: int,
        resume_from: str | Path | None = None,
        on_epoch_end: Callable[[int, float, float], None] | None = None,
    ):
        """Training loop.

        on_epoch_end(epoch, train_loss, val_loss): hook called after validation
        on every epoch. During hyperparameter search it can report the
        intermediate val_loss and raise an exception to prune an unpromising trial.
        """
        start_epoch = self._load_checkpoint(resume_from) if resume_from else 1

        train_size = len(self.train_loader.dataset)
        if self.verbose:
            print(f"\n[Training Start] Total Samples: {train_size}")
            print("-" * 70)

        epoch = start_epoch
        try:
            for epoch in range(start_epoch, epochs + 1):
                train_loss = self._train_epoch(epoch, epochs)
                val_loss = self._validate()

                self.history['train'].append(train_loss)
                self.history['val'].append(val_loss)
                self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']

                if self.verbose:
                    print(f"E{epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.1e}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint("best.pt", epoch, val_loss)
                    if self.verbose:
                        print("  >>> Best Model Saved!")
                else:
                    self.patience_counter += 1

                # Save last.pt every epoch, regardless of early stopping.
                self._save_checkpoint("last.pt", epoch, val_loss)

                # HPO hook: intermediate reporting + pruning (exceptions propagate out of the loop).
                if on_epoch_end is not None:
                    on_epoch_end(epoch, train_loss, val_loss)

                if self.patience_counter >= self.early_stop_patience:
                    if self.verbose:
                        print(f"\n Early stopping at epoch {epoch}")
                    break

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt — saving emergency checkpoint...")
            self._save_checkpoint("emergency_interrupt.pt", epoch, self.best_val_loss)
            print("Emergency checkpoint saved.")
            raise

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        path = self.checkpoint_dir / filename
        model_config = getattr(self.model, 'config', {})

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'history': self.history,
            'config': {'model_args': model_config}
        }, path)

    def _load_checkpoint(self, filepath: str | Path) -> int:
        filepath = Path(filepath)
        if not filepath.exists():
            return 1
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.history = checkpoint.get('history', self.history)
        # Restore best_val_loss from its own key (val_loss is that epoch's loss, not the best).
        self.best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
        self.patience_counter = checkpoint.get('patience_counter', 0)

        print(f"🔄 Resuming from Epoch {checkpoint['epoch'] + 1} "
              f"| best_val_loss={self.best_val_loss:.6f} | patience={self.patience_counter}")
        return checkpoint['epoch'] + 1