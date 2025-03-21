import comet_ml
import os
import torch
import models.BaseModel as BaseModel
import warnings
warnings.filterwarnings("ignore")
from pytorchcv.model_provider import _models as PTCV_MODELS
from argparse import ArgumentParser 
from pytorch_lightning import Trainer
from functools import partial
from utils.misc import MixtureDataset, get_dataset, report_results, ARCH_NAMES
from utils.test_utils import process_dataset, load_model, setup_model_factory
from torch.utils.data import DataLoader
from utils.ConfidenceScores import IMPLEMENTED_CONFIDENCE_SCORES, TEMP_RANGE, WhiteBoxConfidence, BlackBoxConfidence
from third_party.openmix import model_provider as omix_model
import pandas as pd

if __name__ == "__main__":
    supportted_arches = list(PTCV_MODELS.keys())
    supportted_arches.extend(ARCH_NAMES.keys())
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet",
                                          "cifar10c", "imagenetc", "cub", "places365", "svhn"], default='cifar10')
    parser.add_argument("--arch", choices=supportted_arches, default="resnet20_cifar10")
    parser.add_argument("--ood_dataset", choices=["cifar10", "cifar100", "imagenet",
                                          "cifar10c", "imagenetc", "cub", "places365", "svhn"], default=None,
                                          help="Out-of-distribution dataset to mix with main dataset")
    parser.add_argument("--test_batch", type=int, default=128)
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--custom_model", default=None, type=str)
    parser.add_argument("--path", default=None, type=str)
    parser.add_argument("--ckpt", default=None, type=str, help="Path to model checkpoint to load")
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
    parser.add_argument("--tag", type=str, default=None, help="Tag for the comet experiment")


    configs = parser.parse_args()
    exp = comet_ml.Experiment(project_name="RAT-evaluation") if not configs.debug else comet_ml.OfflineExperiment()
    if configs.tag: exp.add_tag(configs.tag)
    exp.log_parameters(vars(configs))
    # Set random seed
    torch.random.manual_seed(configs.seed)

    # Main execution
    dataset, num_classes, preprocess = process_dataset(configs)
    net, preprocess = load_model(configs, preprocess)
    model_factory = setup_model_factory(configs)

    dataset = get_dataset(configs.dataset, is_train=False, preprocess=preprocess)['dataset']
    if configs.ood_dataset:
        ood_dataset = get_dataset(configs.ood_dataset, is_train=False, preprocess=preprocess)['dataset']
        dataset = MixtureDataset(dataset, ood_dataset)

    # Perform multiple experiments
    tester = Trainer(devices=1, default_root_dir=f"./logs/{exp.get_key()}", inference_mode=False)
    tester.logger.log_hyperparams(configs)
    print(net)
    if configs.temp_test is not None:
        test_loader = DataLoader(dataset, batch_size=configs.test_batch, shuffle=False, num_workers=8)
        results_temp = []
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
            for temp in configs.temp_test:
                rr_model = rr_model_factory(net, temp=temp)
                tester.test(rr_model, test_loader)
                rr_results = rr_model.summarize_results()
                results_temp.append({"method": rr_model.score_name,
                                "AURC": rr_results["AURC"][rr_model.score_name],
                                "AUROC": rr_results["AUROC"][rr_model.score_name],
                                "FPR95": rr_results["FPR95"][rr_model.score_name],
                                "temp": temp
                                })
        for temp in configs.temp_test:
            confidence_scores = {score_name: IMPLEMENTED_CONFIDENCE_SCORES[score_name] for score_name in configs.confid_scores}
            for score in confidence_scores.values():
                score.params["temp"] = temp
                score.delta = configs.perturb_level
            model = model_factory(net, confidence_scores, debug=configs.debug)
            tester.test(model, test_loader)
            cf_results = model.summarize_results()
            for name in confidence_scores.keys():
                results_temp.append({"method": name,
                                "AURC": cf_results["AURC"][name],
                                "AUROC": cf_results["AUROC"][name],
                                "FPR95": cf_results["FPR95"][name],
                                "temp": temp
                                })
        csv_save_path = os.path.join(model.logger.log_dir, "metrics_temp.csv")
        final_results = pd.DataFrame(results_temp)
        final_results.to_csv(csv_save_path, index=False)
        exp.log_table(csv_save_path)
        exp.log_parameter("save_dir", model.logger.log_dir)
    else:
        exp_results, exp_params = [], []
        run_times = {}
        for exp_id in range(configs.n_exps):
            confidence_scores = {score_name: IMPLEMENTED_CONFIDENCE_SCORES[score_name] for score_name in configs.confid_scores}
            model = model_factory(net, confidence_scores, debug=configs.debug)
            val_size = int(len(dataset) * configs.val_split)
            val_data, test_data = torch.utils.data.random_split(dataset, lengths=[val_size, len(dataset)-val_size])
            val_loader = DataLoader(val_data, batch_size=configs.test_batch, shuffle=False, num_workers=16, pin_memory=True)
            test_loader = DataLoader(test_data, batch_size=configs.test_batch, shuffle=False, num_workers=16, pin_memory=True)
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
                test_loader = DataLoader(test_data, batch_size=configs.test_batch, shuffle=False, num_workers=16, pin_memory=True)
            # Testing
            results = {"AUROC": {}, "AURC": {}, "FPR95": {}, "RC-Curve": {}}
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
                tester.test(rr_model, test_loader)
                run_times[rr_model.score_name] = rr_model.runtime
                rr_results = rr_model.summarize_results()
                for key in results:
                    results[key].update(rr_results[key])
                params['temp'][rr_model.score_name] = best_temp
                params['delta'][rr_model.score_name] = 0
            if confidence_scores:
                confid_scores_fitted = {}
                for name, confidence_score in model.confidence_scores.items():
                    if isinstance(confidence_score, WhiteBoxConfidence):
                        confidence_score.fit(net, val_loader)
                    else:
                        if configs.perturb_level < 0:
                            confidence_score.delta, _ = confidence_score.fit_perturb(net, val_loader)
                        confidence_score.fit(net, val_loader, perturb=configs.perturb_level)
                model.reset_timer()
                tester.test(model, test_loader)
                cf_results = model.summarize_results()
                run_times.update(model.runtime)
                for key in results:
                    results[key].update(cf_results[key])
                for name, score in model.confidence_scores.items():
                    params['temp'][name]  = score.params['temp'] if 'temp' in score.param_list else 1
                    params['delta'][name]  = score.delta if (configs.perturb_level < 0 and isinstance(score, BlackBoxConfidence))else configs.perturb_level
            exp_results.append(results)
            exp_params.append(params)

        csv_save_path = os.path.join(model.logger.log_dir, "metrics.csv")
        param_save_path = os.path.join(model.logger.log_dir, "params.csv")
        runtime_save_path = os.path.join(model.logger.log_dir, "time.csv")
        param_df = pd.DataFrame.from_dict(exp_params[-1])
        param_df.to_csv(param_save_path)
        print(run_times)
        runtime_df = pd.DataFrame.from_dict(run_times, orient='index', columns=[0])
        runtime_df.to_csv(runtime_save_path)

        final_results = report_results(exp_results, csv_save_path)
        exp.log_table(csv_save_path)
        exp.log_table(param_save_path)
        exp.log_table(runtime_save_path)
        exp.log_parameter("save_dir", model.logger.log_dir)
        for name, curve in exp_results[-1]["RC-Curve"].items():
            exp.log_curve(f"Risk-Coverage-{name}", *zip(*curve))