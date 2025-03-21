'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os
import math
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torchvision.datasets as datasets
import pandas as pd

from itertools import product
from torchvision.datasets import VisionDataset, CIFAR10, CIFAR100
from torchvision import transforms
from torchvision import models as tv_model
from torch.utils.data import ConcatDataset
from collections import defaultdict


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
            'resnet50_v1': (tv_model.ResNet50_Weights.IMAGENET1K_V1, tv_model.resnet50),
            'resnet34': (tv_model.ResNet34_Weights.DEFAULT, tv_model.resnet34),
            'resnet18': (tv_model.ResNet18_Weights.DEFAULT, tv_model.resnet18),
            'resnet101': (tv_model.ResNet101_Weights.DEFAULT, tv_model.resnet101)
}
__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate_AURC(confidence,
                  is_correct,
                  num_evaluate_points=20):
    coverages = []
    risks = []
    residuals = torch.where(is_correct, 0, 1)
    n = len(residuals)
    idx_sorted = torch.argsort(confidence)
    cov = n
    error_sum = torch.sum(residuals[idx_sorted])
    coverages.append(cov/ n)
    risks.append(error_sum.item() / n)
    weights = []
    tmp_weight = 0

    for i in range(0, len(idx_sorted) - 1):
        cov = cov-1
        error_sum = error_sum - residuals[idx_sorted[i]]
        selective_risk = error_sum.item() / (n - 1 - i)
        tmp_weight += 1

        if i == 0 or confidence[idx_sorted[i]] != confidence[idx_sorted[i - 1]]:
            coverages.append(cov / n)
            risks.append(selective_risk)
            weights.append(tmp_weight / n)
            tmp_weight = 0

    if tmp_weight > 0:
        coverages.append(0)
        risks.append(risks[-1])
        weights.append(tmp_weight / n)

    aurc = sum([(risks[i] + risks[i+1]) * 0.5 * weights[i] for i in range(len(weights))])

    curve = (coverages, risks)
    return  aurc, list(zip(*curve))


class CIFARC(VisionDataset):
    def __init__(self, root, corruption_name, transform=None, target_transform=None):
        super(CIFARC, self).__init__(root, transform=transform, target_transform=target_transform)
        
        # Define the corruption path
        self.data_path = os.path.join(self.root, f"{corruption_name}.npy")
        self.label_path = os.path.join(self.root, "labels.npy")
        
        # Load the data and labels
        self.data = np.load(self.data_path)
        self.targets = np.load(self.label_path).tolist()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        # Convert the image to PIL format
        img = transforms.functional.to_pil_image(img)
        if self.transform:
            img = self.transform(img)
            
        if self.target_transform:
            target = self.target_transform(target)
            
        return img, target


def full_cifarc(root, transform=None, target_transform=None, concat=True):
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost',
        'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    datasets = {corruption: CIFARC(root, corruption, transform, target_transform) for corruption in corruptions}
    if concat:
        return ConcatDataset(datasets.values())
    else:
        return datasets



class Cutout(object):
    """Randomly mask out one or more patches from an image.
       https://arxiv.org/abs/1708.04552
    Args:
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        if np.random.choice([0, 1]):
            mask = np.ones((h, w), np.float32)

            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask

        return img


class Imagenet_c(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost',
            'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
            'spatter', 'saturate', 'speckle_noise', 'gaussian_blur'
        ]
        self.levels = [1, 2, 3, 4, 5]
        self.datasets = {corruption: {level: None for level in self.levels} for corruption in self.corruptions}
        for corrution, level in product(self.corruptions, self.levels):
            data_dir = os.path.join(root, corrution, str(level))
            self.datasets[corrution][level] = torchvision.datasets.ImageFolder(data_dir, transform)
        self.dataset_idx = {i: dataset for i, dataset in enumerate(product(self.corruptions, self.levels))}
        self.length_per_dataset = 50000
    def __len__(self):
        return len(self.dataset_idx) * self.length_per_dataset
    
    def __getitem__(self, idx):
        dataset_id = idx // self.length_per_dataset
        img_id = idx % self.length_per_dataset
        corruption, level = self.dataset_idx[dataset_id]
        return self.datasets[corruption][level][img_id]


vision_transforms = {
    "CIFAR10_test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
    "CIFAR10_train": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
    "CIFAR10_train_cutout": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(16)
            ]),
    "Imagenet": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
}

NORMALIZATION_PARAMS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}

def get_dataset(name, is_train, preprocess=None):
    DATASET_ROOT_DIR = os.environ['DATASET_ROOT_DIR']
    if name == 'cifar10c':
        assert not is_train, "CIFAR10C only has test set"
        CIFAR10C_ROOT_DIR = 'corrupt_cifar10/CIFAR-C'
        if preprocess is None:
            preprocess = vision_transforms['CIFAR10_test']
        dataset = full_cifarc(os.path.join(DATASET_ROOT_DIR, CIFAR10C_ROOT_DIR),
                                transform=preprocess)
        print(f"Load dataset CIFAR10C: num of samples={len(dataset)}")
        print(f"Shape of image: {dataset[0][0].shape}, label: {dataset[0][1]}")
        return {
            'num_classes': 10,
            'input_size': 32,
            'dataset': dataset
        }
    elif name == 'cifar10c_split':
        assert not is_train, "CIFAR10C only has test set"
        CIFAR10C_ROOT_DIR = 'corrupt_cifar10/CIFAR-C'
        if preprocess is None:
            preprocess = vision_transforms['CIFAR10_test']
        dataset = full_cifarc(os.path.join(DATASET_ROOT_DIR, CIFAR10C_ROOT_DIR),
                              transform=preprocess,
                              concat=False)
        print(f"Load dataset CIFAR10C: num of samples={len(dataset)}")
        return {
            'num_classes': 10,
            'input_size': 32,
            'dataset': dataset
        }
    elif name == 'cifar10':
        CIFAR10_ROOT_DIR = "cifar10"
        if preprocess is None:
            preprocess = vision_transforms['CIFAR10_train_cutout' if is_train else 'CIFAR10_test']
        dataset = CIFAR10(os.path.join(DATASET_ROOT_DIR, CIFAR10_ROOT_DIR),
                          train=is_train,
                          transform=preprocess)
        return {
            'num_classes': 10,
            'input_size': 32,
            'dataset': dataset
        }
    elif name == 'cifar100':
        CIFAR100_ROOT_DIR = 'cifar100'
        if preprocess is None:
            preprocess = vision_transforms['CIFAR10_train_cutout' if is_train else 'CIFAR10_test']
        dataset = CIFAR100(os.path.join(DATASET_ROOT_DIR, CIFAR100_ROOT_DIR),
                          train=is_train,
                          transform=preprocess, 
                          download=True)
        return {
            'num_classes': 100,
            'input_size': 32,
            'dataset': dataset
        }
    elif name == 'cifar100c':
        assert not is_train, "CIFAR100C only has test set"
        CIFAR100C_ROOT_DIR = 'corrupt_cifar100/CIFAR-C'
        if preprocess is None:
            preprocess = vision_transforms['CIFAR10_test']
        dataset = full_cifarc(os.path.join(DATASET_ROOT_DIR, CIFAR100C_ROOT_DIR),
                                transform=preprocess)
        print(f"Load dataset CIFAR10C: num of samples={len(dataset)}")
        print(f"Shape of image: {dataset[0][0].shape}, label: {dataset[0][1]}")
        return {
            'num_classes': 100,
            'input_size': 32,
            'dataset': dataset
        }
    elif name == 'imagenet':
        IMAGENET_ROOT_DIR = 'imagenet'
        subpath = "train" if is_train else "val"
        if preprocess is None:
            preprocess = vision_transforms['Imagenet']
        dataset = datasets.ImageFolder(
            root=os.path.join(DATASET_ROOT_DIR, IMAGENET_ROOT_DIR, subpath),
            transform=preprocess
        )
        return {
            'num_classes': 1000,
            'input_size': 224,
            'dataset': dataset
        }
    elif name == 'imagenetc':
        IMAGENETC_ROOT_DIR = 'corrupt_imagenet'
        assert not is_train
        if preprocess is None:
            preprocess = vision_transforms['Imagenet']
        dataset = Imagenet_c(
            root=os.path.join(DATASET_ROOT_DIR, IMAGENETC_ROOT_DIR),
            transform=preprocess
        )
        return {
            'num_classes': 1000,
            'input_size': 224,
            'dataset': dataset
        }
    elif name == "places365":
        if preprocess is None:
            preprocess = vision_transforms['Imagenet']
        if is_train:
            try:
                data = datasets.ImageFolder(root=os.path.join(DATASET_ROOT_DIR, 'places365/train'),
                                            transform=preprocess)
            except(RuntimeError):
                data = datasets.Places365(root=os.path.join(DATASET_ROOT_DIR, 'places365'),
                                          split='train-standard', small=True, download=False,
                                          transform=preprocess)
        else:
            try:
                data = datasets.ImageFolder(root=os.path.join(DATASET_ROOT_DIR, 'places365/val_256'),
                                        transform=preprocess)
            except(RuntimeError):
                data = datasets.Places365(root=os.path.join(DATASET_ROOT_DIR, 'places365'),
                                          split='val', small=True, download=False,
                                          transform=preprocess)
        return {
            'num_classes': 365,
            'input_size': 256,
            'dataset': data
        }
    elif name == "cub":
        if preprocess is None:
            preprocess = vision_transforms['Imagenet']
        if is_train:
            data = datasets.ImageFolder(root=os.path.join(DATASET_ROOT_DIR, 'CUB/train'),
                                        transform=preprocess)
        else:
            data = datasets.ImageFolder(root=os.path.join(DATASET_ROOT_DIR, 'CUB/test'),
                                        transform=preprocess)
        return {
            'num_classes': 200,
            'input_size': 224,
            'dataset': data
        }
    elif name == "svhn":
        SVHN_ROOT_DIR = "svhn"
        if preprocess is None:
            preprocess = vision_transforms['Imagenet']
        dataset = datasets.SVHN(root=os.path.join(DATASET_ROOT_DIR, SVHN_ROOT_DIR),
                                split="train" if is_train else "test",
                                transform=preprocess,
                                download=True)
        return {
            'num_classes': 10,
            'input_size': 32,
            'dataset': dataset
        }
    else:
        raise ValueError(f"Dataset {name} is not currently supported by get_datset")


def fpr_at_fixed_tpr(fprs, tprs, thresholds, tpr_level: float = 0.95):
    if all(tprs < tpr_level):
        raise ValueError(f"No threshold allows for TPR at least {tpr_level}.")
    idxs = [i for i, x in enumerate(tprs) if x >= tpr_level]
    if len(idxs) > 0:
        idx = min(idxs)
    else:
        idx = 0
    return fprs[idx], tprs[idx], thresholds[idx]


def get_metric(results, metric):
    metric = metric + "_"
    return {key[len(metric):]: value for key, value in results.items() if key.startswith(metric)}



def report_results(exp_results, csv_save_path):
    METRICS = ("AUROC", "FPR95", "AURC")
    report_results = defaultdict(dict)
    formatters = {"AURC": lambda x: f"{1000*x:.2f}", 
                "AUROC": lambda x: f"{x:.4f}",
                "FPR95": lambda x: f"{x*100:.2f}%"}
    for metric in METRICS:
        exp_results_metric = [result[metric] for result in exp_results] # Filter other metrics
        pd_results = pd.DataFrame(exp_results_metric)
        formatter = formatters[metric]
        for method_name in pd_results.columns:
            metric_mean = pd_results[method_name].mean()
            metric_std = pd_results[method_name].std()
            report_results[method_name][f"{metric}_mean"] = formatter(metric_mean)
            report_results[method_name][f"{metric}_std"] = formatter(metric_std)
    df = pd.DataFrame.from_dict(report_results, orient="index")
    df.rename(columns={'index': "Method"}, inplace=True)
    df.to_csv(csv_save_path)
    return df


class MixtureDataset(torch.utils.data.Dataset):
    def __init__(self, main_dataset, ood_dataset):
        self.main_dataset = main_dataset
        self.ood_dataset = ood_dataset

    def __len__(self):
        return len(self.main_dataset) + len(self.ood_dataset)
    
    def __getitem__(self, idx):
        if idx < len(self.main_dataset):
            return self.main_dataset[idx]
        else:
            img, label = self.ood_dataset[idx - len(self.main_dataset)]
            return img, -1


class RandomMixUp(torch.nn.Module):
    """Randomly apply MixUp to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch, target):
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


class RandomCutMix(torch.nn.Module):
    """Randomly apply CutMix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("Please provide a valid positive value for the num_classes.")
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch, target):
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        _, H, W = F.get_dimensions(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s
