"""
PyTorch 09 — Autograd: Automatic Gradients
==========================================

HOW DOES A NEURAL NETWORK LEARN?
  1. Make a prediction.
  2. Measure how wrong it is (the LOSS).
  3. Compute the GRADIENT: for every weight, "if I increase this weight a
     tiny bit, how does the loss change?"  (the derivative / slope)
  4. Nudge each weight in the direction that REDUCES the loss.

Step 3 by hand is painful. PyTorch's AUTOGRAD does it for you:
  * mark tensors with requires_grad=True
  * do math with them (PyTorch records a "computation graph")
  * call loss.backward()
  * read the gradients from tensor.grad

Run:  python pytorch_basics/09_autograd.py
"""

import torch

# ---------------------------------------------------------------
# 1. The simplest example: y = x^2  ->  dy/dx = 2x
# ---------------------------------------------------------------
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
print(y)            # -> tensor(9., grad_fn=<PowBackward0>)
# grad_fn shows PyTorch remembered HOW y was computed.

y.backward()        # compute dy/dx
print(x.grad)       # -> tensor(6.)    2 * 3 = 6  ✔

# ---------------------------------------------------------------
# 2. More variables: z = w*x + b
# ---------------------------------------------------------------
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(4.0)           # input data: no gradient needed
z = w * x + b                   # 2*4 + 1 = 9
z.backward()
print(w.grad)       # -> tensor(4.)    dz/dw = x = 4
print(b.grad)       # -> tensor(1.)    dz/db = 1
# Reading it: increasing w by 1 increases z by about 4.

# ---------------------------------------------------------------
# 3. Gradient of a loss w.r.t. a weight (a tiny "model")
# ---------------------------------------------------------------
w = torch.tensor(0.5, requires_grad=True)
x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])        # the real rule is y = 2x

y_pred = w * x
loss = ((y_pred - y_true) ** 2).mean()         # mean squared error
print(loss)         # -> tensor(10.5000, grad_fn=<MeanBackward0>)
loss.backward()
print(w.grad)       # -> tensor(-14.)
# Negative gradient => increasing w DECREASES the loss. So w should go up
# (toward 2). Lesson 10 uses this to actually train w.

# ---------------------------------------------------------------
# 4. Gradients ACCUMULATE — you must reset them
# ---------------------------------------------------------------
a = torch.tensor(1.0, requires_grad=True)
for _ in range(3):
    (a * 5).backward()
print(a.grad)       # -> tensor(15.)   5 + 5 + 5, not 5!
a.grad.zero_()      # reset (optimizers do this with optimizer.zero_grad())
print(a.grad)       # -> tensor(0.)

# ---------------------------------------------------------------
# 5. Turning gradient tracking OFF
# ---------------------------------------------------------------
# During evaluation/inference you don't need gradients: turning them off
# saves memory and time.
w = torch.tensor(2.0, requires_grad=True)
with torch.no_grad():
    out = w * 3
print(out.requires_grad)          # -> False

d = w.detach()                    # same value, but cut off from the graph
print(d.requires_grad)            # -> False

# ---------------------------------------------------------------
# 6. Vectors: backward() needs a SCALAR, so reduce first
# ---------------------------------------------------------------
v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
out = (v ** 2).sum()              # sum -> scalar
out.backward()
print(v.grad)       # -> tensor([2., 4., 6.])    d/dv of sum(v^2) = 2v
# (v ** 2).backward() alone -> "grad can be implicitly created only for scalar outputs"

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. For f(x) = 3x^3 + 2x at x = 2, predict the gradient (9x^2 + 2 = 38),
#    then verify with autograd.
# 2. Remove the a.grad.zero_() line and call backward again — what happens?
