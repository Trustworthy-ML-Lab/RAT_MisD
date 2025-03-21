from ..model import resnet18, resnet, densenet_BC, vgg, wrn, efficientnet, mobilenet, convmixer


def get_model(arch, model_dict, num_class):
    if arch == 'resnet18':
        model = resnet18.ResNet18(**model_dict).cuda()
    elif arch == 'res110':
        model = resnet.resnet110(**model_dict).cuda()
    elif arch == 'dense':
        model = densenet_BC.DenseNet3(depth=100, num_classes=num_class,
                                        growth_rate=12, reduction=0.5,
                                        bottleneck=True, dropRate=0.0).cuda()
    elif arch == 'vgg':
        model = vgg.vgg16(**model_dict).cuda()
    elif arch == 'wrn':
        model = wrn.WideResNet(28, num_class, 10).cuda()
    elif arch == 'efficientnet':
        model = efficientnet.efficientnet(**model_dict).cuda()
    elif arch == 'mobilenet':
        model = mobilenet.mobilenet(**model_dict).cuda()
    elif arch == "cmixer":
        model = convmixer.ConvMixer(256, 16, kernel_size=8, patch_size=1, n_classes=num_class).cuda()
    return model