# Title
A disentangled transformer-based transfer learning framework to predict patient drug response from tumor single-cell transcriptomics

## 1 Architecture
<img src="./image.png">

## 2 Requirements
+ Python 3.11
+ CUDA 12.2
+ PyTorch 1.12.1
+ Pandas 2.1.4
+ Numpy 1.26.3
+ Scikit-learn 1.3.0

## 3 Basic Usage

## 3.1 Pre-train encoders
```bash
pretrain_main.py
```

## 3.2 Fine-tune encoders for different drug response and testing on sc data
```bash
drug_ft_main.py
```

