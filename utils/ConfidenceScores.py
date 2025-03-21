import copy
import numpy as np
from numpy.core.multiarray import array as array
import scipy.optimize
import torch
from typing import Callable
from pytorch_lightning import Trainer
from torch.nn.functional import softmax
import torch.utils
from tqdm import tqdm
import scipy 

import models.BaseModel as basemodel
from utils.misc import get_metric


TEMP_RANGE = [0.2, 0.4, 0.6, 0.8, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 2.5, 3, 100, 1000]
DELTA_RANGE = [0, 0.00005, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.0006, 0.0008, 0.001
            #    ,0.0012, 0.0014, 0.0016, 0.0018, 0.002, 0.0022, 0.0024, 0.0026, 0.0028, 0.003, 0.0032, 0.0036, 0.0038, 0.004
            ]
PARAM_SEARCHSPACE = {'temp': TEMP_RANGE,
                     'delta': DELTA_RANGE}

    
class BlackBoxConfidence(Callable):
    param_list = []
    def __call__(self, logits) -> torch.Tensor:
        pass

    def _process(self, x, apply_softmax=True) -> np.array:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if apply_softmax:
            x = softmax(x, dim=1)
        return x
    
    def fit(self, net, val_loader, perturb=0):
        aurcs = self._run_tests(net, {'base': self}, val_loader, perturb=perturb) 
        return aurcs['base']

    @classmethod
    def fit_perturb(cls, *args, perturb_levels=DELTA_RANGE):
        score = cls()
        metrics, scores = {}, {}
        for perturb in perturb_levels:
            metrics[perturb] = score.fit(*args, perturb=perturb)
            scores[perturb] = copy.deepcopy(score)
            if metrics[perturb] is None: return None # When fit method not implemented, return None.
        best_perturb = min(metrics, key=metrics.get)
        print(metrics)
        print(f"Best perturb level: {best_perturb:.4f}")
        return best_perturb, scores[best_perturb]

    def _run_tests(self, net, confidence_scores, val_loader, perturb=0.0):
        if perturb > 0:
            tester = basemodel.NoisePerturbedTester(noise_level=perturb, model=net,
                                                    confid_scores=confidence_scores, plot=False)
            validator = Trainer(devices=1, default_root_dir="./val_logs", inference_mode=False)
        else:
            tester = basemodel.MisDetectTesterBase(net, confidence_scores, plot=False)
            validator = Trainer(devices=1, default_root_dir="./val_logs")
        validator.test(tester, val_loader, verbose=False)[0]
        val_results = tester.summarize_results()
        aurcs = val_results["AURC"]
        return aurcs

class MSR(BlackBoxConfidence):
    def __call__(self, x) -> torch.Tensor:
        x = self._process(x)
        return torch.max(x, dim=1)[0]


class ODIN(BlackBoxConfidence):
    param_list = ['temp']
    def __init__(self, temp=1):
        self.params = {'temp': temp}

    def  __call__(self, x) -> torch.Tensor:
        x = self._process(x)
        return torch.max(x, dim=1)[0]
    
    def _process(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = softmax(x / self.params['temp'], dim=1)
        return x
    
    def fit(self, net, val_loader, perturb=0):
        temp_range = TEMP_RANGE
        confidence_scores = {f"{temp:.2f}": ODIN(temp) for temp in temp_range}
        aurcs = self._run_tests(net, confidence_scores, val_loader, perturb=perturb) 
        best_temp = min(aurcs, key=aurcs.get)
        print(aurcs, "perturb=", perturb)
        self.params['temp'] = float(best_temp)
        print(f"Choosing best temp {self.params['temp']:.2f} after validation")
        return aurcs[best_temp]

class NE(BlackBoxConfidence):
    def __call__(self, x) -> torch.Tensor:
        x = self._process(x)
        return torch.sum(x * torch.log(x + 1e-9), dim=1)
    

class conf_HPS(BlackBoxConfidence):
    def __init__(self, k=2):
        self.k = k
    
    def __call__(self, x) -> torch.Tensor:
        x = self._process(x)
        return -torch.topk(x, 2, dim=1)[0][:, 1]


class mix_HPS_MSR(BlackBoxConfidence):
    def __init__(self, lam=1):
        self.lam = lam

    def __call__(self, x) -> torch.Tensor:
        x = self._process(x)
        sortx, _ = torch.topk(x, 2, dim=1)
        return self.lam * sortx[:, 0] - sortx[:, 1]

    def fit(self, net, val_loader, perturb=0):
        lam_range = [0.01, 0.1, 1, 10, 100]
        confidence_scores = {f"{lam:.2f}": mix_HPS_MSR(lam) for lam in lam_range}
        aurcs = self._run_tests(net, confidence_scores, val_loader, perturb=perturb) 
        best_lam = min(aurcs, key=aurcs.get)
        self.lam = float(best_lam)
        print(f"Choosing best lam {self.lam:.2f} after validation")
        return aurcs[best_lam]


class thresholded_MSR(BlackBoxConfidence):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, x):
        x = self._process(x)
        sortx, _ = torch.topk(x, 2, dim=1)
        return torch.minimum(sortx[:, 0], self.threshold * torch.ones_like(sortx[:, 0])) - sortx[:, 1]

    def fit(self, net, val_loader, perturb=0):
        threshold_range = np.linspace(0.1, 1, 10)
        confidence_scores = {f"{threshold:.2f}": thresholded_MSR(threshold) for threshold in threshold_range}
        aurcs = self._run_tests(net, confidence_scores, val_loader, perturb=perturb)
        best_threshold = min(aurcs, key=aurcs.get)
        self.threshold = float(best_threshold)
        print(f"Choosing best threshold {self.threshold:.2f} after validation")
        return aurcs[best_threshold]
    

class MLS(BlackBoxConfidence):
    def __call__(self, x):
        x = self._process(x, apply_softmax=False)
        return torch.max(x, dim=1)[0]


class TFDoctor(BlackBoxConfidence):
    def __call__(self, x):
        x = self._process(x)
        t = torch.sum(x ** 2, dim=1)
        return -1 / t


class Doctor(BlackBoxConfidence):
    param_list = ['temp']
    def __init__(self, temp=1):
        self.params = {'temp': temp}

    def __call__(self, x):
        x = self._process(x)
        t = torch.sum(x ** 2, dim=1)
        return -1 / t
    
    def _process(self, x) -> np.array:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = softmax(x / self.params['temp'], dim=1)
        return x
    
    def fit(self, net, val_loader, perturb=0):
        temp_range = TEMP_RANGE
        confidence_scores = {f"{temp:.2f}": Doctor(temp) for temp in temp_range}
        aurcs = self._run_tests(net, confidence_scores, val_loader, perturb=perturb) 
        best_temp = min(aurcs, key=aurcs.get)
        self.params['temp'] = float(best_temp)
        print(f"Choosing best temp {self.params['temp']:.2f} after validation")
        return aurcs[best_temp]


class RelU(BlackBoxConfidence):
    param_list = ['temp']
    def __init__(self, train_loader=None, lbd=0.5, temperature=1):
        self.lbd = lbd
        self.params = {'temp': temperature}
        self.weights = None
        self.train_dataloader = train_loader
        self.train_val_split = 0.5

    def _train_model(self, net):
        # get train logits
        train_logits = []
        train_labels = []
        net.eval()
        net.to('cuda')
        for data, labels in self.train_dataloader:
            data = data.to('cuda')
            with torch.no_grad():
                logits = net(data).cpu()
            train_logits.append(logits)
            train_labels.append(labels)
        train_logits = torch.cat(train_logits, dim=0)
        train_pred = train_logits.argmax(dim=1)
        train_labels = torch.cat(train_labels, dim=0)
        train_labels = (train_labels != train_pred).int()

        train_probs = torch.softmax(train_logits / self.params['temp'], dim=1)

        train_probs_pos = train_probs[train_labels == 0]
        train_probs_neg = train_probs[train_labels == 1]
        assert train_probs_neg.numel() > 0, "No negative examples found"
        self.weights = -(1 - self.lbd) * torch.einsum("ij,ik->ijk", train_probs_pos, train_probs_pos).mean(dim=0).to('cuda') \
            + self.lbd * torch.einsum("ij,ik->ijk", train_probs_neg, train_probs_neg).mean(dim=0).to('cuda')
        self.weights = torch.tril(self.weights, diagonal=-1)
        self.weights = self.weights + self.weights.T
        self.weights = torch.relu(self.weights)

        if torch.all(self.weights <= 0):
            # default to gini
            self.weights = torch.ones(self.weights.size()).to('cuda')
            self.weights = torch.tril(self.weights, diagonal=-1)
            self.weights = self.weights + self.weights.T
        self.weights = self.weights / self.weights.norm()

    def __call__(self, logits):
        probs = torch.softmax(logits / self.params['temp'], dim=1)
        weights = torch.tril(self.weights, diagonal=-1)
        weights = weights + weights.T
        weights = weights / weights.norm()
        return -torch.diag(probs @ weights @ probs.T)

    def fit(self, net, val_loader, perturb=0):
        temp_range = TEMP_RANGE
        # Split data
        dataset = val_loader.dataset
        train_size = int(self.train_val_split * len(dataset))
        with torch.random.fork_rng():
            train_idx, val_idx = torch.split(torch.randperm(len(dataset)), [train_size, len(dataset) - train_size])
        train_ds, val_ds = torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, val_idx)
        train_loader = torch.utils.data.DataLoader(train_ds, num_workers=val_loader.num_workers, batch_size=val_loader.batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_ds, num_workers=val_loader.num_workers, batch_size=val_loader.batch_size, shuffle=False)
        confidence_scores = {f"{temp:.2f}": RelU(train_loader=train_loader, temperature=temp) for temp in temp_range}
        # Train the candidate scores
        for score_func in confidence_scores.values():
            score_func._train_model(net)
        aurcs = self._run_tests(net, confidence_scores, val_loader, perturb=perturb) 
        best_temp = min(aurcs, key=aurcs.get)
        self.params['temp'] = float(best_temp)
        self.weights = confidence_scores[best_temp].weights
        print(f"Choosing best temp {self.params['temp']:.2f} after validation")
        return aurcs[best_temp]

    def export_matrix(self):
        return self.weights.cpu()


class ratio(BlackBoxConfidence):    
    def __call__(self, x):
        x = self._process(x)
        top_2 = torch.topk(x, 2, dim=1)[0]
        return top_2[:, 0] / top_2[:, 1]

class LogitDiff(BlackBoxConfidence):
    def __call__(self, x):
        x = self._process(x, apply_softmax=False)
        top_2 = torch.topk(x, 2, dim=1)[0]
        return top_2[:, 0] - top_2[:, 1]


class SoftLogitDiff(BlackBoxConfidence):
    param_list = ['temp']
    def __init__(self, temp=1):
        self.params = {'temp': temp}
    def __call__(self, x):
        x = self._process(x, apply_softmax=False)
        logits, _ = torch.sort(x, dim=-1, descending=True) 
        logits = logits / self.params['temp']
        soft_runnerup = torch.logsumexp(logits[:, 1:], dim=-1)
        return logits[:, 0] - soft_runnerup


class TunedLogitDiff(BlackBoxConfidence):
    param_list = ['logi_diff_lambda']
    def __init__(self, logi_diff_lambda=1):
        self.lam = logi_diff_lambda

    def __call__(self, x):
        x = self._process(x, apply_softmax=False)
        top_2 = torch.topk(x, 2, dim=1)[0]
        return top_2[:, 0] - self.lam * top_2[:, 1]

    def fit(self, net, val_loader, perturb=0):
        lam_range = (0.5, 0.75, 0.9, 1.1, 1.3, 2)
        confidence_scores = {f"{lam:.2f}": TunedLogitDiff(lam) for lam in lam_range}
        aurcs = self._run_tests(net, confidence_scores, val_loader, perturb=perturb) 
        best_lam = min(aurcs, key=aurcs.get)
        self.lam = float(best_lam)
        print(f"Choosing best lam {self.lam:.2f} after validation")
        return aurcs[best_lam]


class WhiteBoxConfidence(Callable):
    param_list = []
    def __call__(self, inputs, net) -> torch.Tensor:
        pass

    def _run_tests(self, net, confidence_scores, val_loader):
        tester = basemodel.MisDetectTesterBase(net, confidence_scores, plot=False)
        validator = Trainer(devices=1, default_root_dir="./val_logs", inference_mode=False)
        validator.test(tester, val_loader, verbose=False)
        val_results = tester.summarize_results()
        aurcs = val_results["AURC"]
        return aurcs

    def fit(self, net, val_loader):
        temp_range = TEMP_RANGE
        confidence_scores = {}
        if 'temp' in self.param_list:
            for temp in temp_range:
                new_score = copy.copy(self)
                new_score.params['temp'] = temp
                confidence_scores[temp] = (new_score)
            aurcs = self._run_tests(net, confidence_scores, val_loader) 
            best_temp = min(temp_range, key=aurcs.get)
            self.params['temp'] = best_temp
            print(f"Choosing best temp {best_temp:.2f} after validation")
            return best_temp


class NoisePerturb(WhiteBoxConfidence):
    def __init__(self, base_score: BlackBoxConfidence, delta=-1):
        self.base_score = base_score
        self.param_list = base_score.param_list
        self.param_list.append('delta')
        self.params = base_score.params
        self.params['delta'] = delta

    def __call__(self, inputs, net) -> torch.Tensor:
        temp_input = torch.autograd.Variable(inputs, requires_grad=True)
        temp_logits = net(temp_input)
        scores = self.base_score(temp_logits)
        loss = torch.log(scores).sum()
        loss.backward()
        modified_input = inputs - self.params['delta'] * torch.sign(temp_input.grad)
        with torch.no_grad():
            final_scores = self.base_score(net(modified_input))
        return final_scores

    def fit(self, net, val_loader):
        metrics = {}
        for perturb in DELTA_RANGE:
            score = copy.deepcopy(self.base_score)
            metrics[perturb] = score.fit(net, val_loader)
            if metrics[perturb] is None: return None # When fit method not implemented, return None.
        best_perturb = min(metrics, key=metrics.get)
        print(f"Best perturb level: {best_perturb:.4f}")
        return best_perturb

class RobustRadiusFast(WhiteBoxConfidence):
    param_list = ['temp']
    def __init__(self, temp=1, step_size=1e-3):
        self.params = {'temp': temp}
        self.step_size = step_size

    def __call__(self, inputs, net) -> torch.Tensor:
        ori_input = torch.autograd.Variable(inputs, requires_grad=True)
        ori_logits = net(ori_input) / self.params['temp']
        ori_pred = torch.softmax(ori_logits, dim=-1)
        scores, _ = ori_pred.max(dim=-1)
        loss = torch.log(scores).sum()
        loss.backward()
        direction = torch.sign(ori_input.grad)
        step_size = self.step_size
        with torch.no_grad():
            modified_input = inputs - step_size * direction
            modified_logits = net(modified_input)
            delta = modified_logits - ori_logits
            max_v, max_idx = ori_logits.max(dim=-1)
            grad = delta - delta[torch.arange(len(delta)), max_idx][:, None]
            grad = grad / step_size
            grad[torch.arange(len(delta)), max_idx] = 1 # Prevent divided by 0
            diff = max_v[:, None] - ori_logits
            perturb_required = diff / grad
            perturb_required[perturb_required <= 0] = 1
            scores, _ = perturb_required.min(dim=-1)
        return scores


class RobustRadiusFastExp(WhiteBoxConfidence):
    def __call__(self, inputs, net) -> torch.Tensor:
        ori_input = torch.autograd.Variable(inputs, requires_grad=True)
        ori_logits = net(ori_input)
        top_logits, _ = ori_logits.topk(k=2, dim=-1)
        logits_diff = top_logits[:, 0] - top_logits[:, 1]
        logits_diff.sum().backward()
        grad = ori_input.grad
        with torch.no_grad():
            scores = logits_diff / grad.abs().reshape((grad.size(0), -1)).sum(dim=-1)
        return scores


class RobustRadiusFastSmooth(WhiteBoxConfidence):
    param_list = ['temp']
    def __init__(self, temp=1):
        self.params = {'temp': temp}

    def __call__(self, inputs, net) -> torch.Tensor:
        ori_input = torch.autograd.Variable(inputs, requires_grad=True)
        ori_logits = net(ori_input) / self.params['temp']
        logits, _ = torch.sort(ori_logits, dim=-1, descending=True)
        soft_runnerup = torch.logsumexp(logits[:, 1:], dim=-1)
        diff = logits[:, 0] - soft_runnerup
        diff.sum().backward()
        with torch.no_grad():
            scores = diff / ori_input.grad.abs().reshape((ori_input.grad.size(0), -1)).sum(dim=-1)
        return scores


class RobustRadiusFastFull(WhiteBoxConfidence):
    def __call__(self, inputs, net) -> torch.Tensor:
        preds = net(inputs)
        scores = []
        for input, pred in inputs, preds:
            jacobian = torch.autograd.functional.jacobian(net, input.unsqueeze(0), vectorize=True)
            target = pred.argmax()
            objective = torch.cat((pred[:target], pred[target:])) - pred[target]
            jacobian_obj = torch.cat((jacobian[:target, :], jacobian[target:, :])) - jacobian[target, :]
            N_targets, N_vars = jacobian_obj.shape
            A_ub_1 = np.concatenate((np.eye(N_vars), -np.ones(N_vars, 1)), axis=1)
            A_ub_2 = np.concatenate((-np.eye(N_vars), -np.ones(N_vars, 1)), axis=1)
            A_ub_3 = np.concatenate((-jacobian_obj.cpu().numpy(), np.zeros(N_targets, 1)), axis=1)
            b_ub_1, b_ub_2 = np.zeros((N_vars,)), np.zeros((N_vars,))
            b_ub_3 = objective.cpu().numpy()
            A_ub, b_ub = np.concatenate((A_ub_1, A_ub_2, A_ub_3), axis=0), np.concatenate((b_ub_1, b_ub_2, b_ub_3), axis=0)
            c = np.array([0] * N_vars + [1])
            res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub)
            scores.append(res.fun)
        return torch.Tensor(scores)



class RobustRadiusExact(WhiteBoxConfidence):
    param_list = ['temp']
    def __init__(self, temp=1, step_size=1e-3):
        self.params = {'temp': temp}
        self.step_size = step_size

    def __call__(self, inputs, net) -> torch.Tensor:
        ori_input = torch.autograd.Variable(inputs, requires_grad=True)
        ori_logits = net(ori_input) / self.params['temp']
        max_v, max_idx = ori_logits.max(dim=-1)
        ori_pred = torch.softmax(ori_logits, dim=-1)
        scores, _ = ori_pred.max(dim=-1)
        loss = torch.log(scores).sum()
        loss.backward()
        direction = torch.sign(ori_input.grad)
        step_size = self.step_size
        with torch.no_grad():
            step_size = torch.autograd.Variable(torch.zeros((ori_input.size(0))), requires_grad=True)
            t = torch.Tensor((0, )).type_as(direction)
            t = torch.autograd.Variable(t)
            grad = torch.autograd.functional.jacobian(lambda x: net(x * direction), t).squeeze()
            grad = grad - grad[torch.arange(len(grad)), max_idx][:, None]
            grad[torch.arange(len(inputs)), max_idx] = 1 # Prevent divided by 0
            diff = max_v[:, None] - ori_logits
            perturb_required = diff / grad
            perturb_required[perturb_required <= 0] = 1
            scores, _ = perturb_required.min(dim=-1)
        return scores

IMPLEMENTED_CONFIDENCE_SCORES = {
    **{'HPS_{:d}'.format(i):conf_HPS(i) for i in range(1, 6)},
    'NE': NE(),
    'MSR': MSR(),
    'HPS': conf_HPS(),
    'linear_mix': mix_HPS_MSR(),
    'thr': thresholded_MSR(),
    'DOCTOR': Doctor(),
    'ODIN': ODIN(),
    'MLS': MLS(),
    'RelU': RelU(),
    'ratio': ratio(),
    'logiDiff': LogitDiff(),
    'softLogiDiff': SoftLogitDiff(),
    'tfDOCTOR': TFDoctor(),
    'tunedLD': TunedLogitDiff(),
    'RR_fast': RobustRadiusFast(),
    'RR_exp': RobustRadiusFastExp(),
    'RR_exact': RobustRadiusExact(),
    'RR_soft': RobustRadiusFastSmooth()
}

IMPLEMENTED_LOGIT_CONFIDENCE_SCORES = {
    'MLS': MLS()
}

if __name__ == "__main__":
    test_array = np.array([[0.3, 0.3, 0.4],
                           [0.1, 0.2, 0.7],
                           [0.9, 0.01, 0.09]])
    for score_name, score in IMPLEMENTED_CONFIDENCE_SCORES.items():
        print(score_name, score(test_array))