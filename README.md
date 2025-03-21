# RAT: Robustness Aware Training for Misclassfication Detection (MisD)
**This is the official repository for the paper ["RAT: Boosting Misclassification Detection Ability without Extra Data"](https://arxiv.org/abs/2503.14783).
The full codebase will be released once we finish the cleanup.**
* We introduce robust radius into the field of Misclassification Detection (MisD) as a confidence score, and design two computation-efficient algorithms: RR-BS and RR-Fast for efficient robust radius estimation.
* We further design a training method called Radius Aware Training (RAT) to boost MisD ability without extra data. 
* We conduct extensive empirical study on our method and several baselines on the task of misclassification detection. Results show that our method outperforms the baselines over most settings, achieving up to 29.3% reduction on AURC and 21.62% reduction in FPR@95TPR.

## Installation
To start, create a conda environment and install the dependencies.
```bash
conda create -n rat python=3.11
conda activate rat
pip install -r requirements.txt
```

## Data preparation
We use CIFAR10, CIFAR100, and ImageNet datasets for training and evaluation.
To prepare the data, set the environment variable `DATASET_ROOT_DIR` to the root directory of the dataset.
The imagenet dataset needs to be downloaded manually and placed in `DATASET_ROOT_DIR/imagenet`, with the following structure:
```
imagenet
├── train
│   ├── n01440764
│   ├── n01443537
│   └── ...
├── val
│   ├── n01440764
│   ├── n01443537
│   └── ...
└── ...
```

## Run RAT training
To run RAT training on CIFAR10 with WRN-28-10 model, use the following command:
```bash
python models/train.py --weight_decay 0.0001 --max_epochs 500 --step_size 0.001 --batch_size 64 --scheduler cosine --ori_lam 1.0 --flex_direction --learning_rate 0.2 --dataset cifar10 --model wrn28_10_cifar10 --from_scratch --optimizer sgd --mixup --mixup_alpha 1.0 --final_epochs_no_mixup 10 --warmup_epochs 5
```
The batch size is batch size per GPU. We use 2 gpus on CIFAR10 and 4 gpus on ImageNet. To reproduce our results,
please set the batch size according to the number of GPUs you have.

## Run evaluation
To evaluate the performance on RAT models, use the following command:
```bash
python main.py --robust_radius --arch wrn28_10_cifar10 --dataset cifar10 --confid_scores MSR ODIN RR_fast --perturb_level -1 --n_exp 3 --data-parallel --ckpt <path_to_checkpoint>
```

### Run evaluation on OpenMix model
If you want to evaluate the performance on OpenMix model, first use their [official repository](https://github.com/Impression2805/OpenMix) to train the model.
Then, put the checkpoint in `third_party/openmix/checkpoints/` with the following structure:
```
openmix
├── checkpoints
│   ├── cifar10
│   │   ├── resnet110.pth
│   │   └── wrn28.pth
│   ├── cifar100
│   │   ├── resnet110.pth
│   │   └── wrn28.pth
```


## Results

![](https://github.com/user-attachments/assets/b36f7c8c-be2d-4511-b27e-0a09773dfbea)
*Figure 1:* MisD results on CIFAR10 dataset.

![](https://github.com/user-attachments/assets/26e5d172-939f-43f1-a4c8-19fd3e2d5007)
*Figure 2:* MisD results on ImageNet dataset.

## Cite our work
```
@article{yan2025rat,
  title={RAT: Boosting Misclassification Detection Ability without Extra Data},
  author={Yan, Ge and Weng, Tsui-Wei},
  journal={arXiv preprint arXiv:2503.14783},
  year={2025}
}
```
