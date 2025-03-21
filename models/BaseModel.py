import time
import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn.functional as F
from functools import partial
from pytorch_lightning.utilities.types import STEP_OUTPUT
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve

import utils.attacks as attacks
from utils.misc import evaluate_AURC, fpr_at_fixed_tpr
from utils.ConfidenceScores import WhiteBoxConfidence


class MisDetectTesterBase(pl.LightningModule):
    def __init__(self, model, confid_scores=[], plot=True, debug=False):
        super(MisDetectTesterBase, self).__init__()
        self.model = model
        self.plot = plot
        self.cache = []
        self.confidence_scores = confid_scores
        self.debug = debug
        self.debug_cache = {}
        self.runtime = defaultdict(lambda: 0.)

    def forward(self, x):
        return self.model(x)
    
    def test_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT | None:
        forward_start_time = time.time()
        x, y = batch
        logits = self.forward(x)
        is_correct = logits.argmax(dim=-1) == y
        forward_time = time.time() - forward_start_time
        self.log("test_acc", is_correct.float().mean(), on_epoch=True, on_step=False)
        # Score calculation
        final_scores = {}
        for confid_score_name, score_func in self.confidence_scores.items():
            method_start_time = time.time()
            if isinstance(score_func, WhiteBoxConfidence):
                with torch.enable_grad():
                    final_scores[confid_score_name] = score_func(x, self.model)
            else:
                final_scores[confid_score_name] = score_func(logits)
            self.runtime[confid_score_name] += time.time() - method_start_time + forward_time
        results = {
            'is_correct': is_correct,
            'y': y
        }
        results.update(final_scores)
        self.cache.append(results)

    def on_test_epoch_start(self) -> None:
        self.cache = []

    def reset_timer(self) -> None:
        self.runtime = defaultdict(lambda: 0.)

    def summarize_results(self):
        is_correct = torch.cat([output['is_correct'] for output in self.cache])
        scores_dict = {name: torch.cat([output[name] for output in self.cache]) 
                       for name in self.confidence_scores}
        if self.debug:
            torch.save({'scores': scores_dict,
                        'correct': is_correct}, self.logger.log_dir + "/results.pth")
        aurcs = {}
        curves = {}
        aurocs, fpr95s = {}, {}
        # Introducing other scoring functions
        for confid_score_name in self.confidence_scores:
            scores = scores_dict[confid_score_name]
            # Calculate AURC curve
            aurc, curve = evaluate_AURC(torch.Tensor(scores), is_correct)
            aurcs[confid_score_name] = aurc
            curves[confid_score_name] = curve
            # Make a plot
            if self.plot:
                plt.plot(*(zip(*curve)), label=confid_score_name) # Truncate the points near 0 coverage which are unstable
            # Check AUROC
            fprs, tprs, thrs = roc_curve(1 - is_correct.cpu().numpy(), -scores.cpu().numpy())
            roc_auc = auc(fprs, tprs)
            fpr95, _, _ = fpr_at_fixed_tpr(fprs, tprs, thrs, 0.95)
            aurocs[confid_score_name] = roc_auc
            fpr95s[confid_score_name] = fpr95
        if self.plot:
            plt.legend()
            plt.savefig("./output.png")
            plt.close('all')
        return {"AURC": aurcs,
                "AUROC": aurocs,
                "FPR95": fpr95s,
                "RC-Curve": curves}


class NoisePerturbedTester(MisDetectTesterBase):
    def __init__(self, *args, noise_level=0.125, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_level = noise_level
        self.runtime = defaultdict(lambda: 0.)
    
    def test_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT | None:
        x, y = batch
        forward_start_time = time.time()
        logits = self.forward(x)
        output = F.softmax(logits,dim=1)
        _, predictions = output.data.max(1)
        is_correct = predictions == y
        forward_time = time.time() - forward_start_time
        final_scores = {}
        for confid_score_name, score_func in self.confidence_scores.items():
            method_start_time = time.time()
            if isinstance(score_func, WhiteBoxConfidence):
                with torch.enable_grad():
                    final_scores[confid_score_name] = score_func(x, self.model)
            else:
                noise_level = self.noise_level
                if hasattr(score_func, 'delta'):
                    noise_level = score_func.delta # Use score-specific delta if specified. 
                with torch.set_grad_enabled(True):
                    temp_input = torch.autograd.Variable(x, requires_grad=True)
                    temp_logits = self.forward(temp_input)
                    scores = score_func(temp_logits)
                    # Do FGSM step
                    loss = torch.log(scores).sum()
                    loss.backward()
                modified_input = x - noise_level * torch.sign(temp_input.grad)
                modified_logits = self.forward(modified_input)
                final_scores[confid_score_name] = score_func(modified_logits)
            self.runtime[confid_score_name] += time.time() - method_start_time + forward_time
        self.log("test_acc", is_correct.float().mean(), on_epoch=True, on_step=False)
        results = {
            'is_correct': is_correct,
            'y': y
        }
        results.update(final_scores)
        self.cache.append(results)


class NoisePerturbedRateTester(MisDetectTesterBase):
    def __init__(self, *args, noise_level_start=1e-6, type='FGSM', fast=False, step_size=1e-2, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_name = "RR_fast"
        self.confidence_scores = [self.score_name]
        self.min_noise = noise_level_start
        self.detect_level = 0
        self.max_iteration = 24
        self.atk_type = type
        self.fast = fast
        self.step_size = step_size
    
    def test_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT | None:
        x, y = batch
        logits = self.forward(x)
        output = F.softmax(logits,dim=1)
        _, predictions = output.data.max(1)
        is_correct = predictions == y
        noise_level_max = torch.inf * torch.ones_like(predictions).float()
        noise_level_min = self.min_noise * torch.ones_like(predictions).float()
        with torch.set_grad_enabled(True):
            temp_input = torch.autograd.Variable(x, requires_grad=True)
            temp_logits = self.forward(temp_input)
            temp_pred = torch.softmax(temp_logits, dim=-1)
            scores, _ = temp_pred.max(dim=-1)
            # Do FGSM step
            loss = torch.log(scores).sum()
            loss.backward()
        if self.atk_type == 'fgsm':
            direction = torch.sign(temp_input.grad)
        elif self.atk_type == 'pgd':
            direction = (temp_input.grad)
            direction = direction / torch.norm(direction, p=torch.inf)
        if self.fast:
            step_size = self.step_size
            modified_input = x - step_size * direction
            modified_logits = self.forward(modified_input)
            delta = modified_logits - temp_logits
            max_v, max_idx = temp_logits.max(dim=-1)
            grad = delta - delta[torch.arange(len(delta)), max_idx][:, None]
            grad = grad / step_size
            grad[torch.arange(len(delta)), max_idx] = 1 # Prevent divided by 0
            diff = max_v[:, None] - temp_logits
            perturb_required = diff / grad
            perturb_required[perturb_required <= 0] = 1
            scores, _ = perturb_required.min(dim=-1)
            err = None
        else:
            if self.atk_type == 'fgsm':
                attack = partial(attacks.FGSM)
            for _ in range(self.max_iteration):
                noise_level = (noise_level_max + noise_level_min) / 2
                noise_level = torch.minimum(noise_level, noise_level_min * 2)
                modified_input = x - noise_level[:, None, None, None] * direction
                modified_logits = self.forward(modified_input)
                new_pred = modified_logits.argmax(dim=-1)
                noise_level_min[new_pred == predictions] = noise_level[new_pred == predictions]
                noise_level_max[new_pred != predictions] = noise_level[new_pred != predictions]
            noise_level_max[torch.isinf(noise_level_max)] = noise_level_min[torch.isinf(noise_level_max)]
            scores = (noise_level_max + noise_level_min) / 2
            err = noise_level_max - noise_level_min 
        self.log("test_acc", is_correct.float().mean(), on_epoch=True, on_step=False)
        results = {
            'is_correct': is_correct,
            'y': y,
            'err': err
        }
        noise_level_max[torch.isinf(noise_level_max)] = noise_level_min[torch.isinf(noise_level_max)]
        results[self.score_name] = scores
        self.cache.append(results)
    
    def on_test_epoch_end(self):
        scores = torch.cat([output[self.score_name] for output in self.cache])
        torch.save(scores, self.logger.log_dir + "/scores.pt")
        is_correct = torch.cat([output['is_correct'] for output in self.cache])
        if self.cache[0]['err'] is not None:
            err = torch.cat([output['err'] for output in self.cache])
            torch.save(err, self.logger.log_dir + "/err.pt")
            torch.save(err / scores, self.logger.log_dir + "/relerr.pt")
        torch.save(is_correct, self.logger.log_dir + "/is_correct.pt")
        if self.plot:
            plt.hist(scores[is_correct].cpu().numpy(), bins=100, alpha=0.5, color='blue')
            plt.hist(scores[~is_correct].cpu().numpy(), bins=100, alpha=0.5, color='orange')
            plt.savefig("./dist.png")
            plt.close('all')


class RobustRadiusTester(NoisePerturbedRateTester):
    def __init__(self, *args, noise_level_start=1e-6, type='FGSM', fast=False, step_size=0.01, temp=1, optim_args=None, targeted=False, **kwargs):
        super().__init__(*args, noise_level_start=noise_level_start, type=type, fast=fast, step_size=step_size, **kwargs)
        self.temp = temp
        self.optim_args = optim_args
        self.score_name = f"RR_{self.atk_type}"
        self.confidence_scores = [self.score_name]
        self.targeted = targeted
        self.runtime = 0.

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT | None:
        start_time = time.time()
        x, y = batch
        logits = self.forward(x)
        output = F.softmax(logits,dim=1)
        _, predictions = output.data.topk(k=2, dim=1)
        predictions, runner_up = predictions[:, 0], predictions[:, 1]
        is_correct = predictions == y
        noise_level_max = torch.inf * torch.ones_like(predictions).float()
        noise_level_min = self.min_noise * torch.ones_like(predictions).float()
        if self.atk_type == 'fgsm':
            attack = lambda lam: attacks.FGSM(model=self.model, eps=lam, temp=self.temp)
        elif self.atk_type == 'pgd':
            attack = lambda lam: attacks.PGD(model=self.model, eps=lam, alpha=lam/4,
                                             steps=self.optim_args['pgd_steps'],
                                             random_start=self.optim_args['pgd_random'],
                                             temp=self.temp)
        elif self.atk_type == 'pgdl2':
            attack = lambda lam: attacks.PGDL2(model=self.model, eps=lam, alpha=lam/5,
                                               steps=self.optim_args['pgd_steps'],
                                               random_start=self.optim_args['pgd_random'],
                                               temp=self.temp)
        elif self.atk_type == 'fgsml2':
            attack = lambda lam: attacks.PGDL2(model=self.model,
                                               eps=lam,
                                               alpha=lam,
                                               steps=1,
                                               random_start=False,
                                               temp=self.temp)
        elif self.atk_type == 'deepfool':
            attack = lambda : attacks.DeepFool(model=self.model,
                                               temp=self.temp)
        elif self.atk_type == 'fab':
            attack = lambda : attacks.FAB(model=self.model,
                                          temp=self.temp,
                                          eps=1)
        if self.atk_type in ('deepfool', 'fab'):
            atk = attack()
            with torch.enable_grad():
                modified_input = atk(x, predictions)
            scores = (x - modified_input).view((x.shape[0], -1)).norm(dim=1, p=torch.inf)
            err = None
        else:
            for _ in range(self.max_iteration):
                noise_level = (noise_level_max + noise_level_min) / 2
                noise_level = torch.minimum(noise_level, noise_level_min * 2)
                atk = attack(noise_level)
                targets = runner_up if self.targeted else None
                with torch.enable_grad():
                    modified_input = atk(x, predictions, target_labels=targets)
                modified_logits = self.forward(modified_input)
                new_pred = modified_logits.argmax(dim=-1)
                noise_level_min[new_pred == predictions] = noise_level[new_pred == predictions]
                noise_level_max[new_pred != predictions] = noise_level[new_pred != predictions]
            noise_level_max[torch.isinf(noise_level_max)] = noise_level_min[torch.isinf(noise_level_max)]
            scores = (noise_level_max + noise_level_min) / 2
            err = noise_level_max - noise_level_min 
        self.runtime += (time.time() - start_time)
        self.log("test_acc", is_correct.float().mean(), on_epoch=True, on_step=False)
        results = {
            'is_correct': is_correct,
            'y': y,
            'err': err
        }
        noise_level_max[torch.isinf(noise_level_max)] = noise_level_min[torch.isinf(noise_level_max)]
        results[self.score_name] = scores
        self.cache.append(results)
