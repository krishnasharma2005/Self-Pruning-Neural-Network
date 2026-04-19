# Self-Pruning Neural Network using Learnable Gates

## Problem Overview

Modern neural networks are often overparameterized, making them inefficient for deployment in real-world systems with memory and computational constraints. A common solution is model pruning, where less important weights are removed after training.

In this project, we implement a self-pruning neural network that learns to prune its own connections during training. Instead of relying on post-processing, the model dynamically identifies and suppresses unimportant weights using learnable gating mechanisms.

## Approach

### 1. Prunable Linear Layer

We designed a custom layer `PrunableLinear`, where each weight is associated with a learnable gate score.

Gate values are computed using:
```plaintext
gate=σ(gate_score)
```
The effective weight becomes:
```plaintext
w~ = w ⋅ gate
```
This allows the network to scale or suppress individual weights during training.

### 2. Hard Pruning Mechanism

Since sigmoid outputs are continuous and rarely reach exact zero, we introduced a threshold-based pruning mechanism:
```plaintext
hard_gate = (gate > 0.6)
```
Weights corresponding to gates below this threshold are effectively removed during the forward pass.

### 3. Straight-Through Estimator (STE)

Hard thresholding is non-differentiable. To allow gradient flow, we used a Straight-Through Estimator (STE):
```plaintext
gate = hard_gate.detach() - gate.detach() + gate
```
This enables:
- Discrete pruning in forward pass
- Continuous gradients during backpropagation

### 4. Sparsity Regularization
To encourage pruning, we added an L1 penalty on gate values:
```
total_loss = CE + λ ⋅ ∑gate
```
Larger λ increases pruning pressure; smaller λ preserves more weights.

## Experimental Setup
dataset: CIFAR-10  
mModel Architecture: Input (3072) → 512 → 256 → 10  
Optimizer: Adam  
Loss Function: Cross-Entropy + Sparsity Loss  
epochs: 10  
default threshold: 0.6.
Final experiments were conducted on Google Colab using GPU acceleration.

## Execution Details

Development and debugging were performed locally using VS Code. Final experiments were executed on Google Colab with GPU acceleration to enable training on the full CIFAR-10 dataset.

## Results
dataset | Lambda | Test Accuracy (%) | Sparsity (%)
test data | --- | --- | ---
0.01 | 41.04 | 49.60%
same as above | ... |
default lambda values and results...
| **Lambda** | **Test Accuracy (%)** | **Sparsity (%)** |
|---|---|---|
| **0.01** | **41.04** | **49.60** |
| **0.05** | **41.52** | **76.86** |
| **0.1** | **39.84** | **80.19** |

## Observations
1. Sparsity vs Lambda:
increasing λ leads to higher sparsity (~50% → ~77% → ~80%), confirming that the regularization term effectively controls pruning.
2. Accuracy vs Sparsity Trade-off:
a accuracy remains relatively stable (~40–41%) even at high sparsity; slight drop at highest λ due to aggressive pruning.
3. Gate Value Distribution:
the distribution shows a dense cluster below (~0.3–0.5), representing pruned connections, and a smaller cluster (~0.7), representing important retained weights.
the separation demonstrates successful learning of useful vs redundant connections.

## Key Insights
defines that neural networks are inherently overparameterized,
enables adaptive pruning during training via learnable gating,
states STE as essential for gradient flow through discrete decisions,
and emphasizes proper threshold calibration (0.6) for effective sparsity,
even with ~80% pruning, performance degradation is minimal.

## Conclusion
the implementation of a self-pruning neural network successfully removes unnecessary weights dynamically during training while maintaining competitive accuracy—highlighting the potential for building efficient neural networks without post-training pruning steps.

## Future Work
extend pruning to convolutional layers,
explore structured pruning (filters/channels),
pursue L0 regularization for exact sparsity,
and benchmark inference speed improvements.
