##  MR-SSDA : "Manifold Reconstructed Semi-Supervised Domain Adaptation for Histopathology Images Classification"

This is the implementation of Manifold Reconstructed Semi-Supervised Domain Adaptation for Histopathology Images Classification in Pytorch.

## Getting Started
### Installation
The code was tested on Ubuntu 18.04.5 under the environment below:
* PyTorch 1.7
* Torchvision
* Python 3.8
* numpy 1.21.2
* tensorflow 2.8.0

## Download Dataset
Download ICIAR_2018 Dataset at https://digestpath2019.grand-challenge.org/Home/.   
Download BreaKHis Dataset at https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/.  
Place it in the directory ./dataset by training set and test set.  
The target domain training set should be divided into labeled and unlabeled parts.

### Train
For example, if you run an experiment on adaptation from ICIAR_2018 to BreakHis,
```
python main.py --source ICIAR_2018 --target BreakHis 
```
