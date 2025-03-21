"""
Trainer for the RadTrainer model.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from torchvision import models as tv_model
from models.RadTrainer import RadTrainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Any, Optional, Tuple
import torchvision.transforms as transforms
from pytorchcv.model_provider import get_model as ptcv_model
from utils.misc import get_dataset, NORMALIZATION_PARAMS


ARCH_NAMES = {
            'vit_b_32': (tv_model.ViT_B_32_Weights.DEFAULT, tv_model.vit_b_32),
            'vit_l_16': (tv_model.ViT_L_16_Weights.DEFAULT, tv_model.vit_l_16),
            'vit_l_32': (tv_model.ViT_L_32_Weights.DEFAULT, tv_model.vit_l_32),
            "vit_b_16": (tv_model.ViT_B_16_Weights.DEFAULT, tv_model.vit_b_16),
            "vit_b_16_swag": (tv_model.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1, tv_model.vit_b_16),
            "wrn_101": (tv_model.Wide_ResNet101_2_Weights.DEFAULT, tv_model.wide_resnet101_2),
            "wrn_101_v1": (tv_model.Wide_ResNet101_2_Weights.IMAGENET1K_V1, tv_model.wide_resnet101_2),
            'densenet201': (tv_model.DenseNet201_Weights.DEFAULT, tv_model.densenet201),
            'densenet161': (tv_model.DenseNet161_Weights.DEFAULT, tv_model.densenet161),
            'efficientnet_b7': (tv_model.EfficientNet_B7_Weights.DEFAULT, tv_model.efficientnet_b7),
            'densenet121': (tv_model.DenseNet121_Weights.IMAGENET1K_V1, tv_model.densenet121),
            'resnet50': (tv_model.ResNet50_Weights.DEFAULT, tv_model.resnet50),
            'resnet152': (tv_model.ResNet152_Weights.DEFAULT, tv_model.resnet152),
            'resnet152_v1': (tv_model.ResNet152_Weights.IMAGENET1K_V1, tv_model.resnet152),
            'resnet50_v1': (tv_model.ResNet50_Weights.IMAGENET1K_V1, tv_model.resnet50),
            'resnet34': (tv_model.ResNet34_Weights.DEFAULT, tv_model.resnet34),
            'resnet18': (tv_model.ResNet18_Weights.DEFAULT, tv_model.resnet18),
            'resnet101': (tv_model.ResNet101_Weights.DEFAULT, tv_model.resnet101)
}
def train_model(model: nn.Module,
                train_dataset: Any,
                val_dataset: Any,
                batch_size: int = 32,
                learning_rate: float = 5e-5,
                weight_decay: float = 1e-4,
                lam: float = 1.0,
                max_epochs: int = 10,
                num_workers: int = 8,
                warmup_epochs: int = 0,
                step_size: float = 0.01,
                ori_lam: float = 1.0,
                flex_direction: bool = False,
                ckpt_path: Optional[str] = None,
                normalization: Tuple[Tuple[float], Tuple[float]] = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                scheduler: Optional[str] = None,
                val_check_interval: int = 2000,
                optimizer: Optional[str] = None,
                pgd: bool = False,
                pgd_steps: int = 7,
                pgd_step_size: float = 0.003,
                mixup: bool = False,
                mixup_alpha: float = 1.0,
                final_epochs_no_mixup: int = 0,
                mixed_precision: bool = False,
                reverse_at: bool = False) -> RadTrainer:
    """
    Finetune a pretrained model using RadTrainer.
    
    Args:
        model: Pretrained model to finetune
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        lam: Lambda weight for radius loss
        max_epochs: Maximum number of training epochs
        num_workers: Number of workers for data loading
        warmup_epochs: Number of epochs for learning rate warmup
        step_size: Step size for adversarial perturbation
        ori_lam: Lambda weight for original loss
        flex_direction: Whether to use flexible direction
        ckpt_path: Path to checkpoint to resume training from
        normalization: Normalization parameters for the model
        scheduler: Scheduler to use
        val_check_interval: Validation check interval
        optimizer: Optimizer to use
        pgd: Whether to use PGD training instead of FGSM
        pgd_steps: Number of PGD steps
        pgd_step_size: Step size for PGD updates (alpha)
        mixup: Whether to use mixup training
        mixup_alpha: Alpha parameter for mixup training
        final_epochs_no_mixup: Number of final epochs to train without mixup
        mixed_precision: Whether to use mixed precision training (FP16)
        reverse_at: Whether to reverse adversarial loss sign during adversarial input computation
    Returns:
        Trained RadTrainer model
    """
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Initialize trainer
    rad_trainer = RadTrainer(
        backbone=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lam=lam,
        warmup_epochs=warmup_epochs,
        ori_lam=ori_lam,
        step_size=step_size,
        flex_direction=flex_direction,
        normalization=normalization,
        scheduler=scheduler,
        optimizer=optimizer,
        pgd=pgd,
        pgd_steps=pgd_steps,
        pgd_step_size=pgd_step_size,
        mixup=mixup,
        mixup_alpha=mixup_alpha,
        final_epochs_no_mixup=final_epochs_no_mixup,
        reverse_at=reverse_at,
    )
    # Setup training
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=-1, # Use all available GPUs
        strategy='ddp', # Distributed data parallel for multi-GPU
        logger=pl.loggers.CometLogger(project_name='rat_training'),
        val_check_interval=val_check_interval if val_check_interval is not None else 1.0,
        precision='bf16-mixed' if mixed_precision else '32', # Enable mixed precision if requested
    )
    
    # Run initial validation
    trainer.validate(rad_trainer, dataloaders=val_loader)
    
    # Train model
    trainer.fit(
        rad_trainer,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path
    )
    
    return rad_trainer


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset to use')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--lam', type=float, default=1.0, help='Lambda weight for radius loss')
    parser.add_argument('--step_size', type=float, default=0.01, help='Step size for adversarial perturbation')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture')
    parser.add_argument('--ori_lam', type=float, default=1.0, help='Lambda weight for original loss')
    parser.add_argument('--flex_direction', action='store_true', help='Use flexible direction')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--val_check_interval', type=int, default=None, help='Validation check interval')
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler to use')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use')
    parser.add_argument('--logging_dir', type=str, default=None, help='Logging directory')
    parser.add_argument('--pgd', action='store_true', 
                   help='Use PGD training instead of FGSM')
    parser.add_argument('--pgd_steps', type=int, default=7,
                   help='Number of PGD steps')
    parser.add_argument('--pgd_step_size', type=float, default=None,
                   help='Step size for PGD updates (alpha)')
    parser.add_argument('--from_scratch', action='store_true', 
                       help='Train model from scratch instead of using pretrained weights')
    parser.add_argument('--warmup_epochs', type=int, default=0, 
                       help='Number of epochs for learning rate warmup')
    parser.add_argument('--mixup', action='store_true', 
                       help='Enable mixup training')
    parser.add_argument('--mixup_alpha', type=float, default=1.0,
                       help='Alpha parameter for mixup training')
    parser.add_argument('--final_epochs_no_mixup', type=int, default=0,
                       help='Number of final epochs to train without mixup')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Enable mixed precision training (FP16)')
    parser.add_argument('--reverse_at', action='store_true',
                       help='Reverse adversarial loss sign during adversarial input computation')
    parser.add_argument('--rand_erase', type=float, default=0.0,
                       help='Random erase probability')
    args = parser.parse_args()
    if args.pgd_step_size is None:
        args.pgd_step_size = args.step_size / 4
    # Initialize model and preprocessing(no normalization)
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        model = ptcv_model(args.model, pretrained=not args.from_scratch)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),])
        val_transform = transforms.ToTensor()
    else:
        if args.model in ARCH_NAMES:
            weights, arch = ARCH_NAMES[args.model]
            preprocess = weights.transforms() if not args.from_scratch else None
            model = arch(weights=weights if not args.from_scratch else None)
        else:
            raise ValueError(f'Model {args.model} not supported')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=args.rand_erase),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    # Load datasets
    train_dataset, val_dataset = get_dataset(args.dataset, is_train=True, preprocess=train_transform)["dataset"], \
        get_dataset(args.dataset, is_train=False, preprocess=val_transform)["dataset"]
    

    # Train model
    trained_model = train_model(
        normalization=NORMALIZATION_PARAMS[args.dataset],
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lam=args.lam,
        ori_lam=args.ori_lam,
        step_size=args.step_size,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
        flex_direction=args.flex_direction,
        ckpt_path=args.ckpt,
        val_check_interval=args.val_check_interval,
        scheduler=args.scheduler,
        optimizer=args.optimizer,
        pgd=args.pgd,
        pgd_steps=args.pgd_steps,
        pgd_step_size=args.pgd_step_size,
        mixup=args.mixup,
        mixup_alpha=args.mixup_alpha,
        final_epochs_no_mixup=args.final_epochs_no_mixup,
        mixed_precision=args.mixed_precision,
        reverse_at=args.reverse_at,
    )
