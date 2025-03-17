# RAT_MisD

# RAT: Robustness Aware Training for Misclassfication Detection (MisD)
This repository contains the code for the paper "RAT: Boosting Misclassification Detection Ability without Extra Data".
The full codebase will be released once we finish the cleanup.

## Abstract
As deep neural networks (DNNs) become increasingly prevalent, particularly in safety-critical 
    areas such as autonomous driving and healthcare, the ability to detect misclassifications 
    is vital. In this work, we explore the task of misclassification detection by leveraging an 
    adversarial-perturbation view: we introduce the <strong>robust radius</strong> (also called input-space margin) 
    as a powerful confidence metric. We propose two efficient estimation algorithms, <strong>RR-BS</strong> 
    and <strong>RR-Fast</strong>, to measure robust radius in practice. We further design a novel training 
    framework called <strong>Radius Aware Training (RAT)</strong> to enhance the model's ability to separate 
    correct and incorrect predictions, all without needing extra data. Our experiments show that 
    RAT substantially reduces misdetection rates compared to existing methods, achieving up to a 
    29.3% improvement in AURC and 21.6% reduction in false-positive rate at 95% true-positive rate 
    (FPR@95TPR) across various benchmarks.

## Results

![](https://github.com/user-attachments/assets/b36f7c8c-be2d-4511-b27e-0a09773dfbea)
*Figure 1:* MisD results on CIFAR10 dataset.

![](https://github.com/user-attachments/assets/26e5d172-939f-43f1-a4c8-19fd3e2d5007)
*Figure 2:* MisD results on ImageNet dataset.
