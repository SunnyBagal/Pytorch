"""
PyTorch 14 — Complete Project: Linear Regression
================================================

Putting everything together in the standard PyTorch workflow:

  1. Data      -> make/load tensors, split into train and test
  2. Model     -> subclass nn.Module
  3. Loss + optimizer
  4. Training loop:  forward -> loss -> zero_grad -> backward -> step
  5. Evaluate  -> model.eval() + torch.inference_mode()
  6. Predict on new data

Task: predict house price (in $1000s) from size (in 100 m^2).
Hidden true rule: price = 50 * size + 30 (+ noise)

Run:  python pytorch_basics/14_linear_regression.py
"""

import torch
from torch import nn

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
# (We stay on CPU/CUDA here: tiny models are often slower on MPS.)

# ---------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------
N = 200
X = torch.rand(N, 1) * 3                      # sizes between 0 and 3  -> shape (200, 1)
y = 50 * X + 30 + torch.randn(N, 1) * 5       # prices with noise      -> shape (200, 1)

# 80/20 train/test split — the test set checks how well we GENERALIZE
split = int(0.8 * N)
X_train, y_train = X[:split].to(device), y[:split].to(device)
X_test, y_test = X[split:].to(device), y[split:].to(device)
print(len(X_train), len(X_test))              # -> 160 40

# ---------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)         # learns one weight and one bias

    def forward(self, x):
        return self.linear(x)

model = LinearRegression().to(device)

# ---------------------------------------------------------------
# 3. Loss and optimizer
# ---------------------------------------------------------------
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# ---------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------
epochs = 500
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)                   # 1. forward
    loss = loss_fn(y_pred, y_train)           # 2. loss
    optimizer.zero_grad()                     # 3. reset grads
    loss.backward()                           # 4. backprop
    optimizer.step()                          # 5. update

    if epoch % 100 == 0 or epoch == epochs - 1:
        model.eval()
        with torch.inference_mode():          # like no_grad(), even faster
            test_loss = loss_fn(model(X_test), y_test)
        print(f"epoch {epoch:3d} | train loss {loss.item():8.2f} | test loss {test_loss.item():8.2f}")

# -> epoch   0 | train loss 12531.72 | test loss  5408.16
# -> epoch 100 | train loss    20.20 | test loss    18.05
# -> epoch 200 | train loss    20.18 | test loss    18.12
# -> epoch 300 | train loss    20.18 | test loss    18.13
# -> epoch 400 | train loss    20.18 | test loss    18.13
# -> epoch 499 | train loss    20.18 | test loss    18.13
# The loss levels off around 20 instead of reaching 0: that's the random noise
# we added (std 5 -> variance ~25). No model can predict pure noise.

# ---------------------------------------------------------------
# 5. What did it learn?
# ---------------------------------------------------------------
w = model.linear.weight.item()
b = model.linear.bias.item()
print(f"learned: price = {w:.2f} * size + {b:.2f}   (true: 50 * size + 30)")
# -> learned: price = 49.94 * size + 30.79   (true: 50 * size + 30)

# ---------------------------------------------------------------
# 6. Predict for new houses
# ---------------------------------------------------------------
new_sizes = torch.tensor([[1.0], [2.5]], device=device)
model.eval()
with torch.inference_mode():
    preds = model(new_sizes)
for s, p in zip(new_sizes.squeeze(1).tolist(), preds.squeeze(1).tolist()):
    print(f"size {s:.1f} -> predicted price ${p:.1f}k")
# -> size 1.0 -> predicted price $80.7k
# -> size 2.5 -> predicted price $155.6k

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Replace SGD with Adam(lr=0.5). Does it converge in fewer epochs?
# 2. Make the data non-linear (y = 10 * X**2 + 5) and see how badly a
#    straight line fits. Then try an MLP from lesson 11.
