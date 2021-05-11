# Kaggle-Breast-Cancer-supervised
semi-supervised-convolustional experiment on Breast Cancer histopathology imagry
## Introduction
This repository contains autoencoder clustering model for Breast Cancer histopathology imagry diagnoses, The models are trained with L2 loss.

Uses Breast Cancer histopathology imagry dataset from https://www.kaggle.com/paultimothymooney/breast-histopathology-images
## Dependencies
```
Pytorch 1.4.0+
cuda 9.2+
python 3.5+
```
## Training with a single GPU
```
python3 train.py
parameter
    --train True/False help = "specify whether train the model or not (default: False)"
    --model_dir help = "Directory containing params.json (default: local directory)"
    --resume help = 'resume from latest checkpoint (default: True)'
    --network help='select network to train on. leave it blank means train on all model'
    --log help='set logging level (default: warning)'
```


for scripted training group, run '''sh train.sh''' for comparing different loss function, run '''sh train_gamma.sh''' for ablation study regarding different gamma value for F-ECE loss.


## Network of Choice
Baseline:
    https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
