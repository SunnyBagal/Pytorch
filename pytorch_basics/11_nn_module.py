"""
PyTorch 11 — Building Models with torch.nn
==========================================

`torch.nn` contains building blocks for neural networks.

  nn.Module     -> base class for ALL models and layers (lesson 12 of basics:
                   you inherit from it and call super().__init__())
  nn.Linear     -> fully connected layer: y = x @ W.T + b
  nn.ReLU       -> activation: max(0, x). Adds NON-linearity so the network
                   can learn curves, not just straight lines.
  nn.Sequential -> stack layers in order, no class needed

A model defines:
  __init__  -> create the layers (the learnable parameters live here)
  forward   -> how input flows through the layers
Then you call model(x) — NOT model.forward(x).

Run:  python pytorch_basics/11_nn_module.py
"""

import torch
from torch import nn

torch.manual_seed(0)

# ---------------------------------------------------------------
# 1. A single Linear layer
# ---------------------------------------------------------------
layer = nn.Linear(in_features=3, out_features=2)   # 3 inputs -> 2 outputs
print(layer)                     # -> Linear(in_features=3, out_features=2, bias=True)
print(layer.weight.shape)        # -> torch.Size([2, 3])   (out, in)
print(layer.bias.shape)          # -> torch.Size([2])
print(layer.weight.requires_grad)  # -> True   parameters track gradients automatically

x = torch.rand(4, 3)             # batch of 4 samples, 3 features each
print(layer(x).shape)            # -> torch.Size([4, 2])   one output row per sample

# ---------------------------------------------------------------
# 2. Activation functions
# ---------------------------------------------------------------
z = torch.tensor([-2.0, -0.5, 0.0, 1.5])
print(nn.ReLU()(z))              # -> tensor([0.0000, 0.0000, 0.0000, 1.5000])
print(torch.sigmoid(z))          # -> tensor([0.1192, 0.3775, 0.5000, 0.8176])  squashes to (0, 1)
print(torch.tanh(z))             # -> tensor([-0.9640, -0.4621,  0.0000,  0.9051])  squashes to (-1, 1)

# ---------------------------------------------------------------
# 3. A custom model class (the standard way)
# ---------------------------------------------------------------
class MLP(nn.Module):            # MLP = multi-layer perceptron
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()                       # ALWAYS first
        self.fc1 = nn.Linear(in_dim, hidden)     # layers assigned to self are
        self.act = nn.ReLU()                     # registered automatically
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.fc1(x)          # (N, in_dim)  -> (N, hidden)
        x = self.act(x)
        x = self.fc2(x)          # (N, hidden)  -> (N, out_dim)
        return x

model = MLP(in_dim=4, hidden=8, out_dim=3)
print(model)
# -> MLP(
# ->   (fc1): Linear(in_features=4, out_features=8, bias=True)
# ->   (act): ReLU()
# ->   (fc2): Linear(in_features=8, out_features=3, bias=True)
# -> )

out = model(torch.rand(5, 4))    # 5 samples in -> 5 rows of 3 scores out
print(out.shape)                 # -> torch.Size([5, 3])

# ---------------------------------------------------------------
# 4. Inspecting parameters
# ---------------------------------------------------------------
for name, p in model.named_parameters():
    print(f"{name:10s} {tuple(p.shape)}")
# -> fc1.weight (8, 4)
# -> fc1.bias   (8,)
# -> fc2.weight (3, 8)
# -> fc2.bias   (3,)

total = sum(p.numel() for p in model.parameters())
print("total parameters:", total)   # -> total parameters: 67   (32+8+24+3)

# ---------------------------------------------------------------
# 5. nn.Sequential — the quick way for simple stacks
# ---------------------------------------------------------------
seq = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 3),
)
print(seq(torch.rand(5, 4)).shape)   # -> torch.Size([5, 3])
print(seq[0])                        # -> Linear(in_features=4, out_features=8, bias=True)

# ---------------------------------------------------------------
# 6. train() vs eval() mode
# ---------------------------------------------------------------
# Some layers (Dropout, BatchNorm) behave differently in training vs testing.
model.train()                        # training mode (default)
print(model.training)                # -> True
model.eval()                         # evaluation mode
print(model.training)                # -> False
# Always call model.eval() before evaluating and model.train() before training.

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Build an MLP for 28x28 images with 10 classes: 784 -> 128 -> 64 -> 10.
#    How many parameters does it have?
# 2. Add nn.Dropout(0.2) after the ReLU and print the model.
