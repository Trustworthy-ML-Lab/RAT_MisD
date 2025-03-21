from collections import defaultdict
import comet_ml
import os
import torch
import models.BaseModel as BaseModel
import warnings
warnings.filterwarnings("ignore")
import torchvision as tv
from pytorchcv.model_provider import get_model as ptcv_model, _models as PTCV_MODELS
from torchvision import models as tv_model
from argparse import ArgumentParser 
from pytorch_lightning import Trainer
from functools import partial
from torchvision.transforms import Normalize
from utils.misc import get_dataset, report_results, vision_transforms, CIFARC
from torch.utils.data import DataLoader
from utils.ConfidenceScores import IMPLEMENTED_CONFIDENCE_SCORES, TEMP_RANGE, WhiteBoxConfidence, BlackBoxConfidence
from third_party.openmix import model_provider as omix_model
from third_party.augmix import model_provider as augmix_model
from third_party.RobustAdversarialNetwork import model_provider as pgd_model
import pandas as pd


ARCH_NAMES = {
            'vit_b_32': (tv_model.ViT_B_32_Weights.DEFAULT, tv_model.vit_b_32),
            'vit_l_16': (tv_model.ViT_L_16_Weights.DEFAULT, tv_model.vit_l_16),
            'vit_l_32': (tv_model.ViT_L_32_Weights.DEFAULT, tv_model.vit_l_32),
            "vit_b_16": (tv_model.ViT_B_16_Weights.DEFAULT, tv_model.vit_b_16),
            "vit_b_16_swag": (tv_model.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1, tv_model.vit_b_16),
            "wrn_101": (tv_model.Wide_ResNet101_2_Weights.DEFAULT, tv_model.wide_resnet101_2),
            "wrn_101_v1": (tv_model.Wide_ResNet101_2_Weights.IMAGENET1K_V1, tv_model.wide_resnet101_2),
            'densenet201': (tv_model.DenseNet201_Weights.DEFAULT, tv_model.densenet201),
            'densenet121': (tv_model.DenseNet121_Weights.IMAGENET1K_V1, tv_model.densenet121),
            'resnet50': (tv_model.ResNet50_Weights.DEFAULT, tv_model.resnet50),
            'resnet50_v1': (tv_model.ResNet50_Weights.IMAGENET1K_V1, tv_model.resnet50),
            'resnet34': (tv_model.ResNet34_Weights.DEFAULT, tv_model.resnet34),
            'resnet18': (tv_model.ResNet18_Weights.DEFAULT, tv_model.resnet18)
}

supportted_arches = list(PTCV_MODELS.keys())
supportted_arches.extend(ARCH_NAMES.keys())
parser = ArgumentParser()
parser.add_argument("--dataset", choices=["cifar10c_split", "imagenetc"], default='cifar10c_split')
parser.add_argument("--arch", choices=supportted_arches, default="resnet20_cifar10")
parser.add_argument("--test_batch", type=int, default=128)
parser.add_argument("--plot", action='store_true')
parser.add_argument("--custom_model", default=None, type=str)
parser.add_argument("--path", default=None, type=str)
parser.add_argument("--confid_scores", nargs='+', default=['MSR'])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--perturb_level", type=float, default=0., help="Noise perturbation level, 0 to disable, -1 for validation search")
parser.add_argument("--val_split", type=float, default=0.2, help="validation split percentage")
parser.add_argument("--n_exps", default=1, type=int)
parser.add_argument("--robust_radius", action='store_true', help='Run robust radius score')
parser.add_argument("--debug", action='store_true')
parser.add_argument("--min_noise", default=1e-6, type=float)
parser.add_argument("--atk_type", default='fgsm', type=str)
parser.add_argument("--targeted", action='store_true')
parser.add_argument("--pgd_steps", default=10, type=int)
parser.add_argument("--pgd_disable_random", action='store_true')
parser.add_argument("--data-parallel", action='store_true')
parser.add_argument("--temp-test", nargs='+', type=float, default=None)
parser.add_argument("--prob-thres",  type=float, default=0.)
configs = parser.parse_args()
exp = comet_ml.Experiment(project_name="Failure-Detection-NeurIPS24") if not configs.debug else comet_ml.OfflineExperiment()
exp.log_parameters(vars(configs))
# Set random seed
torch.random.manual_seed(configs.seed)
if configs.perturb_level != 0:
    model_factory = partial(BaseModel.NoisePerturbedTester,
                            noise_level=configs.perturb_level)
else:
    model_factory = BaseModel.MisDetectTesterBase

dataset = get_dataset(configs.dataset, is_train=False)
num_classes = dataset['num_classes']
if configs.dataset.startswith('cifar'):
    preprocess = vision_transforms["CIFAR10_test"]
else:
    preprocess = vision_transforms["Imagenet"]
if configs.custom_model:
    if configs.custom_model == "openmix":
        net, preprocess = omix_model('cifar10')
    elif configs.custom_model == "augmix":
        assert configs.dataset in ["imagenet", "imagenetc"], "Only imagenet model supported. "
        net = augmix_model()
    elif configs.custom_model == "pgd":
        net = pgd_model()
else:
    if configs.dataset not in ['imagenet', 'imagenetc']:
        # Use ptcv models for CIFAR, SVHN
        net = ptcv_model(configs.arch, pretrained=True)
    else:
        weights, arch = ARCH_NAMES[configs.arch]
        preprocess = weights.transforms()
        net = arch(weights=weights)
if configs.data_parallel:
    net = torch.nn.DataParallel(net)
# Move normalization before the net to enable adversarial
if configs.dataset.startswith('cifar') or isinstance(preprocess, tv.transforms.Compose):
    # For cifar datasets, defaults transforms is an composed transform with last step normalization
    preprocess.transforms, normalization = preprocess.transforms[:-1], preprocess.transforms[-1]
elif configs.dataset.startswith('imagenet'):
    # For ImageNet datasets, pytorch ImageClassification preprocess is used. 
    mean, std = preprocess.mean, preprocess.std
    normalization = Normalize(mean, std)
    preprocess.mean, preprocess.std = [0] * 3, [1] * 3
if configs.custom_model != 'pgd': net = torch.nn.Sequential(normalization, net)
dataset = get_dataset(configs.dataset, is_train=False, preprocess=preprocess)
clean_ds = get_dataset('cifar10' if configs.dataset.startswith('cifar') else 'imagenet', is_train=False, preprocess=preprocess)
# Perform multiple experiments
tester = Trainer(devices=1, default_root_dir=f"./logs/{exp.get_key()}", inference_mode=False)
tester.logger.log_hyperparams(configs)

exp_results, exp_params = [], []
for exp_id in range(configs.n_exps):
    confidence_scores = {score_name: IMPLEMENTED_CONFIDENCE_SCORES[score_name] for score_name in configs.confid_scores}
    model = model_factory(net, confidence_scores, debug=configs.debug)
    val_size = int(len(clean_ds['dataset']) * configs.val_split)
    indices = torch.randperm(len(clean_ds['dataset']))
    val_indices, test_indices = indices[:val_size], indices[val_size:]
    val_data = torch.utils.data.Subset(clean_ds['dataset'], val_indices)
    test_datasets = {corruption: torch.utils.data.Subset(ds, test_indices) for corruption, ds in dataset['dataset'].items()}
    val_loader = DataLoader(val_data, batch_size=configs.test_batch, shuffle=False, num_workers=8)
    test_loaders = {corruption: DataLoader(test_data, batch_size=configs.test_batch, shuffle=False, num_workers=8)
                    for corruption, test_data in test_datasets.items()}
    if configs.prob_thres > 0:
        probs = []
        net = net.cuda()
        net.eval()
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.cuda()
                logits = net(x)
                prob = torch.softmax(logits, dim=-1)
                probs.append(prob)
        probs = torch.cat(probs)
        indices = torch.where(probs.max(dim=1).values >= configs.prob_thres)[0]
        test_data = torch.utils.data.Subset(test_data, indices.cpu())
        print(f"Selected test data num: {len(test_data)}")
        test_loader = DataLoader(test_data, batch_size=configs.test_batch, shuffle=False, num_workers=8)
    # Testing
    results = defaultdict(lambda : {"AUROC": {}, "AURC": {}, "FPR95": {}, "RC-Curve": {}})
    params = {"temp":{}, "delta":{}}
    if configs.robust_radius:
        rr_model_factory = partial(BaseModel.RobustRadiusTester,
                        debug=configs.debug,
                        type=configs.atk_type,
                        fast=False,
                        step_size=0,
                        noise_level_start=configs.min_noise,
                        targeted=configs.targeted, 
                        optim_args={"pgd_steps": configs.pgd_steps,
                                    "pgd_random": not configs.pgd_disable_random})
        aurcs = {}
        for temp in TEMP_RANGE:
            rr_model = rr_model_factory(net, temp=temp)
            tester.test(rr_model, val_loader, verbose=False)
            aurcs[temp] = rr_model.summarize_results()['AURC'][rr_model.score_name]
        best_temp = min(TEMP_RANGE, key=lambda x:aurcs[x])
        print(f"Best temp for {rr_model.score_name}: {best_temp:.2f}")
        rr_model = rr_model_factory(net, temp=best_temp)
        for corruption, test_loader in test_loaders.items():
            tester.test(rr_model, test_loader)
            rr_results = rr_model.summarize_results()
            for key in results[corruption]:
                results[corruption][key].update(rr_results[key])
            params['temp'][rr_model.score_name] = best_temp
            params['delta'][rr_model.score_name] = 0
    if confidence_scores:
        for confidence_score in model.confidence_scores.values():
            if isinstance(confidence_score, WhiteBoxConfidence):
                confidence_score.fit(net, val_loader)
            else:
                if configs.perturb_level < 0:
                    confidence_score.delta = confidence_score.fit_perturb(net, val_loader)
                confidence_score.fit(net, val_loader, perturb=configs.perturb_level)
        for corruption, test_loader in test_loaders.items():
            tester.test(model, test_loader)
            cf_results = model.summarize_results()
            for key in results[corruption]:
                results[corruption][key].update(cf_results[key])
            for name, score in model.confidence_scores.items():
                params['temp'][name]  = score.params['temp'] if 'temp' in score.param_list else 1
                params['delta'][name]  = score.delta if (configs.perturb_level < 0 and isinstance(score, BlackBoxConfidence))else configs.perturb_level
    exp_results.append(results)
    exp_params.append(params)

param_save_path = os.path.join(model.logger.log_dir, "params.csv")
param_df = pd.DataFrame.from_dict(exp_params[-1])
param_df.to_csv(param_save_path)
for corruption in exp_results[0]:
    os.makedirs(os.path.join(model.logger.log_dir, corruption))
    csv_save_path = os.path.join(model.logger.log_dir, corruption, "metric.csv")
    all_exp_results = [exp_result[corruption] for exp_result in exp_results]
    final_results = report_results(all_exp_results, csv_save_path)
    exp.log_table(csv_save_path)
exp.log_parameter("save_dir", model.logger.log_dir)