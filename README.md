# RAT: Robustness Aware Training for Misclassfication Detection (MisD)
**This is the official repository for the paper ["RAT: Boosting Misclassification Detection Ability without Extra Data"](https://arxiv.org/abs/2503.14783).
The full codebase will be released once we finish the cleanup.**
* We introduce robust radius into the field of Misclassification Detection (MisD) as a confidence score, and design two computation-efficient algorithms: RR-BS and RR-Fast for efficient robust radius estimation.
* We further design a training method called Radius Aware Training (RAT) to boost MisD ability without extra data. 
* We conduct extensive empirical study on our method and several baselines on the task of misclassification detection. Results show that our method outperforms the baselines over most settings, achieving up to 29.3% reduction on AURC and 21.62% reduction in FPR@95TPR.

## Results

![](https://github.com/user-attachments/assets/b36f7c8c-be2d-4511-b27e-0a09773dfbea)
*Figure 1:* MisD results on CIFAR10 dataset.

![](https://github.com/user-attachments/assets/26e5d172-939f-43f1-a4c8-19fd3e2d5007)
*Figure 2:* MisD results on ImageNet dataset.
