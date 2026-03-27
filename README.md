# Title
A disentangled transformer-based transfer learning framework to predict patient drug response from tumor single-cell transcriptomics

## 1 Architecture
<img src="./image.png">

## 2 Requirements
+ Python 3.9.9
+ CUDA 11.3
+ PyTorch 1.12.0
+ Pandas 1.5.3
+ Numpy 1.26.0
+ Scikit-learn 1.2.1
+ Scanpy 1.10.3
+ Scipy 1.13.1

## 3 Basic Usage

## 3.1 Pre-train encoders
```bash
pretrain_main.py
```

## 3.2 Fine-tune encoders for different drug response and testing on sc data
```bash
drug_ft_main.py
```

