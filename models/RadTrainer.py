"""
Model trainer for maximizing robust radius of correct predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Optional, Tuple
from pytorch_lightning.utilities.types import STEP_OUTPUT
from math import cos, pi


STEP_RATIO = 1.25

class RadTrainer(pl.LightningModule):
    def __init__(self, backbone: nn.Module,
                 learning_rate: float = 5e-5,
                 weight_decay: float = 1e-4,
                 lam: float = 1.0,
                 ori_lam: float = 1.0,
                 step_size: float = 1e-3,
                 flex_direction: bool = False,
                 normalization: Tuple[Tuple[float], Tuple[float]] = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 scheduler: Optional[str] = None,
                 optimizer: Optional[str] = None,
                 pgd: bool = False,
                 pgd_steps: int = 7,
                 pgd_step_size: float = 0.003,
                 warmup_epochs: int = 0,
                 mixup: bool = False,
                 mixup_alpha: float = 1.0,
                 final_epochs_no_mixup: int = 0,
                 reverse_at: bool = False):
        super().__init__()
        self.backbone = backbone
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lam = lam
        self.ori_lam = ori_lam
        self.step_size = step_size
        self.flex_direction = flex_direction
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.validation_step_outputs = []
        # ImageNet normalization parameters
        self.register_buffer('mean', torch.tensor(normalization[0]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(normalization[1]).view(1, 3, 1, 1))
        self.pgd = pgd
        self.pgd_steps = pgd_steps
        self.pgd_step_size = pgd_step_size
        self.warmup_epochs = warmup_epochs
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.final_epochs_no_mixup = final_epochs_no_mixup
        self.reverse_at = reverse_at
        
    def normalize(self, x):
        return (x - self.mean) / self.std
        
    def forward(self, x):
        # Normalize input before passing to backbone
        x = self.normalize(x)
        logits = self.backbone(x)
        return logits
        
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        x.requires_grad = True

        # Check if we should use mixup based on remaining epochs
        use_mixup = self.mixup and self.training
        if self.final_epochs_no_mixup > 0:
            epochs_remaining = self.trainer.max_epochs - self.current_epoch
            use_mixup = use_mixup and epochs_remaining > self.final_epochs_no_mixup

        if use_mixup:
            x_mix, y_a, y_b, lam = self.mixup_data(x, y)
            logits_mix = self.forward(x_mix)
            loss = self.mixup_criterion(F.cross_entropy, logits_mix, y_a, y_b, lam)
            # For accuracy calculation with mixup
            logits = self.forward(x)
            pred = logits.argmax(dim=-1)
            acc = pred.eq(y).float().mean()
        else:
            logits = self.forward(x)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(dim=-1) == y).float().mean()

        is_correct = logits.argmax(dim=-1) == y
        
        if self.pgd:
            # PGD attack
            x_adv = x.clone().detach()
            
            for _ in range(self.pgd_steps):
                x_adv.requires_grad = True
                logits_adv = self.forward(x_adv)
                loss_adv = F.cross_entropy(logits_adv, y)
                grad = torch.autograd.grad(loss_adv, x_adv, retain_graph=False, create_graph=False)[0]

                with torch.no_grad():
                    if self.reverse_at:
                        grad = -grad

                    if self.flex_direction:
                        x_adv_new = x_adv.clone()
                        # Move in opposite directions based on correctness
                        x_adv_new[is_correct] = x_adv[is_correct] + self.pgd_step_size * grad[is_correct].sign()
                        x_adv_new[~is_correct] = x_adv[~is_correct] - self.pgd_step_size * grad[~is_correct].sign()
                        x_adv = x_adv_new
                    else:
                        x_adv = x_adv + self.pgd_step_size * grad

                    # Project back to epsilon ball
                    delta = x_adv - x
                    delta = torch.clamp(delta, -self.step_size, self.step_size)
                    x_adv = x + delta
                    x_adv = torch.clamp(x_adv, 0, 1)

            adv_x = x_adv
        else:
            # Original FGSM logic
            grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
            if self.reverse_at:
                grad = -grad
            if self.flex_direction:
                adv_x = x.clone()
                adv_x[is_correct] = x[is_correct] + self.step_size * grad[is_correct].sign()
                adv_x[~is_correct] = x[~is_correct] - self.step_size * grad[~is_correct].sign()
            else:
                adv_x = x + self.step_size * grad.sign()

        # Rest of the training step remains the same
        adv_logits = self.forward(adv_x)
        adv_loss = F.cross_entropy(adv_logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_adv_loss', adv_loss)
        self.log('train_acc', acc)
        
        loss = loss * self.ori_lam + adv_loss * self.lam
        return loss
    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        
        # Track predictions and loss across validation set
        correct = (logits.argmax(dim=-1) == y).sum()
        total = len(y)
        # Log running metrics
        self.validation_step_outputs.append({
            'correct': correct,
            'total': total,
            'loss': loss.item() * total
        })
        
        return loss

    def on_validation_epoch_end(self):
        # Gather outputs from all GPUs
        outputs = self.validation_step_outputs
        
        if self.trainer.world_size > 1:
            # Gather all metrics across GPUs
            gathered_correct = self.all_gather(torch.Tensor([x['correct'] for x in outputs]))
            gathered_total = self.all_gather(torch.Tensor([x['total'] for x in outputs]))
            gathered_loss = self.all_gather(torch.Tensor([x['loss'] for x in outputs]))
            
            # Sum across all GPUs - gathered tensors are of shape [num_gpus, num_batches]
            total_correct = gathered_correct.sum().item()
            total_samples = gathered_total.sum().item()
            total_loss = gathered_loss.sum().item()
        else:
            # Single GPU case
            total_correct = sum(x['correct'] for x in outputs)
            total_samples = sum(x['total'] for x in outputs)
            total_loss = sum(x['loss'] for x in outputs)
        
        val_acc = total_correct / total_samples
        val_loss = total_loss / total_samples
        
        # Log validation metrics (Lightning automatically handles DDP logging)
        if self.trainer.is_global_zero:  # Only print on main process
            print(f"Epoch {self.current_epoch} Validation accuracy: {val_acc}, Validation loss: {val_loss}")
        self.log('val_acc', val_acc, sync_dist=True)
        self.log('val_loss', val_loss, sync_dist=True)
        
        # Clear saved outputs
        self.validation_step_outputs = []
        
    def on_train_epoch_start(self):
        """Log learning rate at the start of each epoch"""
        # Get current learning rate from optimizer
        opt = self.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        
        # Log the learning rate
        current_lr = opt.param_groups[0]['lr']
        self.log('learning_rate', current_lr)

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )

        if self.scheduler == 'cosine':
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs - self.warmup_epochs
            )
            
            if self.warmup_epochs > 0:
                rel_lr = 1e-8 / self.learning_rate
                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda epoch: rel_lr + (1 - rel_lr) * (cos(pi + pi * epoch / self.warmup_epochs) + 1) / 2
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[self.warmup_epochs]
                )
                return [optimizer], [scheduler]
            return [optimizer], [main_scheduler]
        else:
            if self.warmup_epochs > 0:
                rel_lr = 1e-8 / self.learning_rate
                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda epoch: rel_lr + (1 - rel_lr) * (cos(pi + pi * epoch / self.warmup_epochs) + 1) / 2
                )
                constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer,
                    factor=1.0,
                    total_iters=self.trainer.max_epochs - self.warmup_epochs
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, constant_scheduler],
                    milestones=[self.warmup_epochs]
                )
                return [optimizer], [scheduler]
            return optimizer

    def on_save_checkpoint(self, checkpoint):
        # Only save backbone state dict
        checkpoint['state_dict'] = {
            k.replace('backbone.', ''): v 
            for k, v in checkpoint['state_dict'].items()
            if k.startswith('backbone.')
        }
    
    def on_load_checkpoint(self, checkpoint):
        # Convert state dict back to backbone format
        checkpoint['state_dict'] = {
            f'backbone.{k}': v
            for k, v in checkpoint['state_dict'].items()
        }
        checkpoint['state_dict']['mean'] = self.mean
        checkpoint['state_dict']['std'] = self.std

    def mixup_data(self, x: torch.Tensor, y: torch.Tensor):
        """Performs mixup on the input and returns mixed inputs, pairs of targets, and lambda"""
        if self.mixup_alpha > 0:
            # Sample lambda from beta distribution
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        
        # Generate permuted indices
        index = torch.randperm(batch_size).to(x.device)

        # Perform mixup on both images and labels
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Mixup loss function"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
