"""
PyTorch 13 — Optimizers (torch.optim)
=====================================

In lesson 10 we updated weights by hand:  w -= lr * w.grad
An OPTIMIZER does this for ALL of a model's parameters with 2 calls:

    optimizer.zero_grad()   # reset old gradients
    loss.backward()         # compute new gradients
    optimizer.step()        # update every parameter

Popular optimizers:
  SGD   -> plain gradient descent (optionally with momentum)
  Adam  -> adapts the step size per parameter; great default choice
  AdamW -> Adam with better weight decay; common for modern models

Run:  python pytorch_basics/13_optimizers.py
"""

import torch
from torch import nn

# ---------------------------------------------------------------
# 1. One optimizer step, observed closely
# ---------------------------------------------------------------
torch.manual_seed(0)
layer = nn.Linear(1, 1)
optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

print(f"before: w={layer.weight.item():.4f}")   # -> before: w=-0.0075

x = torch.tensor([[2.0]])
y = torch.tensor([[10.0]])
loss = nn.MSELoss()(layer(x), y)

optimizer.zero_grad()
loss.backward()
print(f"grad:   {layer.weight.grad.item():.4f}")  # -> grad:   -37.9141
optimizer.step()                                 # w = w - 0.1 * grad
print(f"after:  w={layer.weight.item():.4f}")    # -> after:  w=3.7839

# ---------------------------------------------------------------
# 2. Compare SGD vs Adam on the same problem (learn y = 3x + 2)
# ---------------------------------------------------------------
X = torch.linspace(-1, 1, 50).unsqueeze(1)       # shape (50, 1)
Y = 3 * X + 2

def train_with(opt_class, **opt_kwargs):
    torch.manual_seed(0)                          # identical starting weights
    model = nn.Linear(1, 1)
    opt = opt_class(model.parameters(), **opt_kwargs)
    loss_fn = nn.MSELoss()
    for _ in range(50):
        loss = loss_fn(model(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()

print(f"SGD            final loss: {train_with(torch.optim.SGD, lr=0.1):.5f}")
print(f"SGD + momentum final loss: {train_with(torch.optim.SGD, lr=0.1, momentum=0.9):.5f}")
print(f"Adam           final loss: {train_with(torch.optim.Adam, lr=0.1):.5f}")
# -> SGD            final loss: 0.00273
# -> SGD + momentum final loss: 0.02277
# -> Adam           final loss: 0.01502
# Different optimizers converge at different speeds. No single one always wins —
# here plain SGD happens to win; momentum overshoots a bit on this easy problem.
# On real, bigger networks Adam is usually the safest first choice.

# ---------------------------------------------------------------
# 3. Learning-rate schedulers — change lr during training
# ---------------------------------------------------------------
model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=1.0)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)  # halve every 2 epochs
lrs = []
for epoch in range(6):
    opt.step()                    # (normally: a full epoch of training here)
    scheduler.step()              # call once per epoch, AFTER optimizer.step()
    lrs.append(opt.param_groups[0]["lr"])
print(lrs)                        # -> [1.0, 0.5, 0.5, 0.25, 0.25, 0.125]

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Run train_with(torch.optim.SGD, lr=1.2). What happens? Why?
# 2. Try torch.optim.AdamW(lr=0.1, weight_decay=0.01).
