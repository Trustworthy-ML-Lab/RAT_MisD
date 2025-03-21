import torch
import torchvision as tv
from torchvision.transforms import Normalize
from functools import partial
from models import BaseModel
from pytorchcv.model_provider import get_model as ptcv_model

from third_party.openmix import model_provider as omix_model
from utils.misc import vision_transforms, get_dataset, ARCH_NAMES


def process_dataset(configs):
    dataset = get_dataset(configs.dataset, is_train=False)
    num_classes = dataset['num_classes']
    if configs.dataset in ('cifar10c', 'cifar100c'):
        preprocess = vision_transforms["CIFAR10_test"]
    elif configs.dataset == 'imagenetc':
        preprocess = vision_transforms["Imagenet"]
    else:
        preprocess = dataset['dataset'].transform
    return dataset, num_classes, preprocess

def load_model(configs, preprocess):
    if configs.custom_model:
        if configs.custom_model == "openmix":
            net, preprocess = omix_model(configs.dataset)
    else:
        if configs.dataset not in ['imagenet', 'imagenetc']:
            # Use ptcv models for CIFAR, SVHN
            net = ptcv_model(configs.arch, pretrained=True)
        else:
            if configs.arch in ARCH_NAMES:
                weights, arch = ARCH_NAMES[configs.arch]
                preprocess = weights.transforms()
                net = arch(weights=weights)
            else:
                net = ptcv_model(configs.arch, pretrained=True)
        
    
    # Load checkpoint if specified
    if configs.ckpt is not None:
        print(f"Loading checkpoint from {configs.ckpt}")
        checkpoint = torch.load(configs.ckpt)
        net.load_state_dict(checkpoint['state_dict'])
    
    if configs.data_parallel:
        net = torch.nn.DataParallel(net)
    # Move normalization before the net to enable adversarial
    if configs.dataset in ('cifar10', 'cifar100', 'cifar10c') or isinstance(preprocess, tv.transforms.Compose):
        # For cifar datasets, defaults transforms is an composed transform with last step normalization
        preprocess.transforms, normalization = preprocess.transforms[:-1], preprocess.transforms[-1]
    elif configs.dataset in ('imagenet', 'imagenet_c'):
        # For ImageNet datasets, pytorch ImageClassification preprocess is used. 
        mean, std = preprocess.mean, preprocess.std
        normalization = Normalize(mean, std)
        preprocess.mean, preprocess.std = [0] * 3, [1] * 3
    
    if configs.custom_model != 'pgd':
        net = torch.nn.Sequential(normalization, net)
    
    return net, preprocess

def setup_model_factory(configs):
    if configs.perturb_level != 0:
        return partial(BaseModel.NoisePerturbedTester, noise_level=configs.perturb_level)
    else:
        return BaseModel.MisDetectTesterBase
