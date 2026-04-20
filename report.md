#  Self-Pruning Neural Network using Learnable Gates

---

##  Problem Overview

Modern neural networks are often overparameterized, making them inefficient for deployment in real-world systems with memory and computational constraints. Model pruning is commonly used to reduce model size by removing less important weights.

In this project, we implement a **self-pruning neural network** that learns to remove its own unnecessary connections during training. Instead of post-training pruning, the model dynamically suppresses unimportant weights using learnable gating mechanisms.

---

##  Approach

### 1. Learnable Gating Mechanism

Each weight in the network is associated with a learnable **gate score**.

The gate value is computed using a sigmoid function:

gate = sigmoid(gate_score)

where:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

The effective weight becomes:

$$
\tilde{w}_{ij} = w_{ij} \cdot \text{gate}_{ij}
$$


This allows the model to scale or suppress individual weights during training.

---

### 2. Hard Pruning

Since sigmoid outputs are continuous and do not reach exact zero, we apply threshold-based pruning:

**hard_gate = 1 if gate > 0.6 else 0**

Weights below this threshold are effectively removed during the forward pass.

---

### 3. Straight-Through Estimator (STE)

Hard thresholding is non-differentiable. To enable gradient flow, we use a Straight-Through Estimator (STE):

**gate = hard_gate.detach() - gate.detach() + gate**

This allows:
- Discrete pruning during forward pass  
- Gradient flow during backpropagation  

---

### 4. Loss Function

The model is trained using a combination of classification loss and sparsity regularization.

**Total Loss = CrossEntropy Loss + λ × Sparsity Loss**

Where:

Sparsity Loss = sum of all gate values

The hyperparameter λ controls the strength of pruning.

---

##  Experimental Setup

- Dataset: CIFAR-10  
- Model Architecture:
  - Input (3072) → 512 → 256 → 10  
- Optimizer: Adam  
- Number of Epochs: 10  
- Pruning Threshold: 0.6  

---

##  Results

| Lambda | Test Accuracy (%) | Sparsity (%) |
|--------|------------------|--------------|
| 0.01   | 41.04            | 49.60        |
| 0.05   | 41.52            | 76.86        |
| 0.1    | 39.84            | 80.19        |

---

##  Observations

### 1. Effect of Sparsity Coefficient

Increasing λ leads to higher sparsity:

- λ = 0.01 → ~50% sparsity  
- λ = 0.05 → ~77% sparsity  
- λ = 0.1 → ~80% sparsity  

This confirms that the sparsity regularization term effectively controls pruning.

---

### 2. Accuracy vs Sparsity Trade-off

- Accuracy remains relatively stable (~40–41%) even at high sparsity  
- A slight drop is observed at higher λ due to aggressive pruning  

This indicates that a large portion of the model parameters are redundant.

---

### 3. Sparsity Calculation

**Sparsity (%) = (Number of pruned weights / Total number of weights) × 100**

---

### 4. Gate Value Distribution

<img width="640" height="480" alt="gate_distribution" src="https://github.com/user-attachments/assets/909184a3-e9ea-469e-952a-1fbc6c4f7b08" />


The gate value distribution shows:

- A large concentration of low-value gates → pruned weights  
- A smaller cluster of higher values → important retained weights  

Although the values are not strictly binary, thresholding enables effective separation between active and inactive connections.

---

##  Key Insights

- Neural networks are inherently **overparameterized**  
- Learnable gating enables **dynamic pruning during training**  
- The Straight-Through Estimator (STE) is essential for enabling gradient flow through discrete pruning operations  
- Proper threshold selection is critical for achieving meaningful sparsity  
- High sparsity (~80%) can be achieved with minimal accuracy loss  

---

##  Conclusion

We successfully implemented a self-pruning neural network that dynamically removes unnecessary weights during training. The model achieves high sparsity levels while maintaining competitive accuracy, demonstrating the effectiveness of combining learnable gating, sparsity regularization, and gradient approximation techniques.

This approach highlights the potential for building efficient neural networks without requiring post-training pruning.

---

##  Future Work

- Extend pruning to convolutional layers  
- Explore structured pruning (filters or channels)  
- Apply L0 regularization for exact sparsity  
- Benchmark inference speed improvements  

---
