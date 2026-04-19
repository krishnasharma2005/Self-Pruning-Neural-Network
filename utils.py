import torch
from model import PrunableLinear

def compute_sparsity(model, threshold=0.6):
    total = 0
    zero = 0

    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = torch.sigmoid(layer.gate_scores)
            total += gates.numel()
            zero += (gates < threshold).sum().item()

    return 100 * zero / total


def compute_sparsity_loss(model):
    loss = 0
    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = torch.sigmoid(layer.gate_scores)
            loss += 5 * gates.mean()
    return loss