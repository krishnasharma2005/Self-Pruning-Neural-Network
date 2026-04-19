# Self-Pruning Neural Network

A deep learning project focused on **automatic model compression using pruning techniques**, enabling efficient neural networks with reduced parameters while maintaining performance.

---

##  Overview

This project implements a **self-pruning neural network** that dynamically removes less important weights during training. The goal is to:

* Reduce model size
* Improve inference efficiency
* Maintain competitive accuracy

This approach is particularly useful for **edge devices and resource-constrained environments**.

---

##  Features

* Automatic weight pruning during training
* Supports configurable pruning thresholds
* Works with standard datasets (e.g., CIFAR-10)
* Modular and extensible architecture
* Clean training pipeline with logging

---

##  Project Structure

```
self_pruning_nn/
│
├── data/                 # Dataset directory (ignored in Git)
├── models.py               # Model architectures
├── train.py             # Training scripts
├── utils.py                # Helper functions
├── requirements.txt      # Dependencies
└── README.md
```

---

##  Installation

### 1. Clone the repository

```
git clone https://github.com/krishnasharma2005/Self-Pruning-Neural-Network.git
cd Self-Pruning-Neural-Network
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

##  Dataset

This project uses the **CIFAR-10 dataset**.

⚠️ Dataset is not included in the repository due to size limits.

It will be automatically downloaded during execution, or you can manually download it from:

https://www.cs.toronto.edu/~kriz/cifar.html

---

##  Usage

Run training:

```
python train.py
```

You can modify hyperparameters such as:

* Learning rate
* Pruning threshold
* Number of epochs

inside the configuration files or script.

---

##  Methodology

The model applies **iterative pruning** during training:

1. Train the network normally
2. Identify low-magnitude weights
3. Prune (set to zero) those weights
4. Fine-tune the network

This results in a **sparse and efficient model**.

---

##  Results

* Reduced number of parameters
* Maintained competitive accuracy
* Improved efficiency for deployment

<img width="597" height="455" alt="image" src="https://github.com/user-attachments/assets/444a55b9-61d7-416c-b3f2-2f95e543bb63" />



---

##  Future Work

* Structured pruning (filters / channels)
* Integration with quantization
* Deployment on edge devices
* Visualization dashboard

---


##  License

This project is open-source and available under the MIT License.


---
