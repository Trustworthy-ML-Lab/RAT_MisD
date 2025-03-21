import torch
import os
import torchvision as tv

from .utils.model_utils import get_model


CIFAR10_WEIGHTS = {'resnet': os.path.join(os.path.dirname(__file__), "checkpoints/cifar10/resnet110.pth"),
                   'wrn': os.path.join(os.path.dirname(__file__), "checkpoints/cifar10/wrn28.pth")}
CIFAR100_WEIGHTS = {'resnet': os.path.join(os.path.dirname(__file__), "checkpoints/cifar100/resnet110.pth"),
                   'wrn': os.path.join(os.path.dirname(__file__), "checkpoints/cifar100/wrn28.pth")}

class truncated_base(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)[..., :-1]
    

def model_provider(dataset, arch='wrn'):
    model_name = {'wrn': 'wrn',
                  'resnet': 'res110'}
    if dataset == 'cifar10':
        model = get_model(model_name[arch], {'num_classes': 11}, num_class=11)
        ckpt = torch.load(CIFAR10_WEIGHTS[arch])
        model.load_state_dict(ckpt)
        transform =  tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                    std=[0.247, 0.243, 0.262]),
        ])
    elif dataset == 'cifar100':
        model = get_model(model_name[arch], {'num_classes': 101}, num_class=101)
        ckpt = torch.load(CIFAR100_WEIGHTS[arch])
        model.load_state_dict(ckpt)
        transform =  tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                    std=[0.267, 0.256, 0.276]),
        ])
    truncated_model = truncated_base(model)
    return truncated_model, transform