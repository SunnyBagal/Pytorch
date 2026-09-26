"""
PyTorch 10 — Gradient Descent by Hand
=====================================

Now we TRAIN something using only tensors + autograd — no nn, no optim.
Understanding this loop means you understand deep learning training.

Task: learn y = 3x + 2 from noisy data. The model is  y_pred = w*x + b.
We start with random w, b and let gradient descent find w≈3, b≈2.

GRADIENT DESCENT:  weight = weight - learning_rate * gradient
  * the gradient points "uphill" (toward more loss)
  * so we step the OPPOSITE way
  * learning_rate controls the step size

Run:  python pytorch_basics/10_gradient_descent_manual.py
"""

import torch

torch.manual_seed(0)

# ---------------------------------------------------------------
# 1. Make a fake dataset
# ---------------------------------------------------------------
X = torch.linspace(-1, 1, 100)                  # 100 inputs
y = 3 * X + 2 + 0.1 * torch.randn(100)          # true rule + a bit of noise

# ---------------------------------------------------------------
# 2. Parameters to learn (random start)
# ---------------------------------------------------------------
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
print(f"start: w={w.item():.3f}, b={b.item():.3f}")
# -> start: w=1.323, b=0.000

learning_rate = 0.1

# ---------------------------------------------------------------
# 3. The training loop
# ---------------------------------------------------------------
for epoch in range(100):
    # (a) FORWARD pass: make predictions
    y_pred = w * X + b

    # (b) LOSS: mean squared error
    loss = ((y_pred - y) ** 2).mean()

    # (c) BACKWARD pass: compute d(loss)/dw and d(loss)/db
    loss.backward()

    # (d) UPDATE: step against the gradient. Wrapped in no_grad because the
    #     update itself should not be recorded in the computation graph.
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # (e) RESET gradients (they accumulate otherwise — lesson 09)
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 20 == 0 or epoch == 99:
        print(f"epoch {epoch:3d} | loss {loss.item():.4f} | w {w.item():.3f} | b {b.item():.3f}")

# -> epoch   0 | loss 4.9814 | w 1.437 | b 0.401
# -> epoch  20 | loss 0.0682 | w 2.618 | b 1.985
# -> epoch  40 | loss 0.0139 | w 2.906 | b 2.004
# -> epoch  60 | loss 0.0107 | w 2.977 | b 2.004
# -> epoch  80 | loss 0.0105 | w 2.994 | b 2.004
# -> epoch  99 | loss 0.0105 | w 2.998 | b 2.004

print(f"learned: y = {w.item():.2f}x + {b.item():.2f}   (true: y = 3x + 2)")
# -> learned: y = 3.00x + 2.00   (true: y = 3x + 2)

# The loss went DOWN and w, b moved toward 3 and 2. That's learning!
#
# Everything later is this same loop with conveniences:
#   nn.Module   -> holds w, b for you          (lesson 11)
#   nn.MSELoss  -> the loss formula            (lesson 12)
#   optim.SGD   -> the update + zero_grad      (lesson 13)

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Set learning_rate = 1.5. What happens to the loss? (It explodes: too big a step.)
# 2. Set learning_rate = 0.001. How many epochs do you need now?
# 3. Change the true rule to y = -2x + 5 and check the model learns it.
