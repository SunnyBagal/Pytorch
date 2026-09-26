"""
PyTorch 16 — Complete Project: Neural Network Classifier
========================================================

A full, realistic training script — the template you'll reuse for most
projects. Task: classify 2-D points into 3 classes arranged in a SPIRAL.
A straight line can't separate spirals, so we need a non-linear network
(Linear + ReLU layers).

Workflow:
  data -> Dataset/DataLoader -> model -> loss + optimizer
       -> train loop (per epoch: train, then validate) -> final evaluation

Run:  python pytorch_basics/16_classification_nn.py
"""

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------
# 1. Data: 3 interleaved spiral arms (300 points each)
# ---------------------------------------------------------------
def make_spirals(n_per_class=300, n_classes=3, noise=0.2):
    X, y = [], []
    for c in range(n_classes):
        r = torch.linspace(0.0, 1.0, n_per_class)                        # radius
        theta = torch.linspace(c * 4, (c + 1) * 4, n_per_class) + torch.randn(n_per_class) * noise
        X.append(torch.stack([r * torch.sin(theta), r * torch.cos(theta)], dim=1))
        y.append(torch.full((n_per_class,), c))
    return torch.cat(X), torch.cat(y)

X, y = make_spirals()
print(X.shape, y.shape, y.dtype)   # -> torch.Size([900, 2]) torch.Size([900]) torch.int64

dataset = TensorDataset(X, y)
train_ds, val_ds = random_split(dataset, [720, 180])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# ---------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------
class SpiralNet(nn.Module):
    def __init__(self, hidden=64, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),      # outputs raw LOGITS (no softmax)
        )

    def forward(self, x):
        return self.net(x)

model = SpiralNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---------------------------------------------------------------
# 3. Reusable train / evaluate functions
# ---------------------------------------------------------------
def train_one_epoch(model, loader):
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(xb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        count += len(xb)
    return total_loss / count, correct / count


@torch.no_grad()                       # decorator form of `with torch.no_grad():`
def evaluate(model, loader):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item() * len(xb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        count += len(xb)
    return total_loss / count, correct / count

# Before training: accuracy should be about 1/3 (random guessing)
_, acc0 = evaluate(model, val_loader)
print(f"before training: val acc {acc0:.1%}")    # -> before training: val acc 34.4%

# ---------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------
epochs = 60
history = {"train_loss": [], "val_loss": [], "val_acc": []}
for epoch in range(1, epochs + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader)
    va_loss, va_acc = evaluate(model, val_loader)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)
    if epoch == 1 or epoch % 10 == 0:
        print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.1%} "
              f"| val loss {va_loss:.3f} acc {va_acc:.1%}")

# -> epoch  1 | train loss 0.858 acc 53.6% | val loss 0.599 acc 64.4%
# -> epoch 10 | train loss 0.057 acc 97.9% | val loss 0.030 acc 99.4%
# -> epoch 20 | train loss 0.026 acc 98.9% | val loss 0.014 acc 100.0%
# -> epoch 30 | train loss 0.018 acc 99.4% | val loss 0.006 acc 100.0%
# -> epoch 40 | train loss 0.036 acc 98.6% | val loss 0.042 acc 98.3%
# -> epoch 50 | train loss 0.023 acc 99.0% | val loss 0.010 acc 100.0%
# -> epoch 60 | train loss 0.016 acc 99.7% | val loss 0.003 acc 100.0%
# Loss doesn't fall perfectly smoothly (see epoch 40) — small bumps are normal.

# ---------------------------------------------------------------
# 5. Predict new points
# ---------------------------------------------------------------
model.eval()
new_points = torch.tensor([[0.0, 0.1], [0.5, -0.5]], device=device)
with torch.inference_mode():
    probs = torch.softmax(model(new_points), dim=1)
preds = probs.argmax(dim=1)
for p, cls, pr in zip(new_points.tolist(), preds.tolist(), probs.tolist()):
    print(f"point {[round(v, 2) for v in p]} -> class {cls}  probs {[round(v, 2) for v in pr]}")
# -> point [0.0, 0.1] -> class 0  probs [1.0, 0.0, 0.0]
# -> point [0.5, -0.5] -> class 0  probs [1.0, 0.0, 0.0]
# (Exact numbers depend on training; the predicted class is what matters.)

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Remove the ReLUs (keep only Linear layers). Accuracy drops a lot —
#    without non-linearity the whole network is just one straight-line model.
# 2. Try hidden=8 and hidden=256. Watch train vs val accuracy (overfitting?).
# 3. Add nn.Dropout(0.1) after each ReLU.
