"""
PyTorch 04 — Tensor Operations
==============================

Neural networks are built from a small set of operations:
  * element-wise math  (+, -, *, /)
  * matrix multiplication (@)  <- the MOST important one
  * aggregations (sum, mean, max, argmax)

Run:  python pytorch_basics/04_tensor_operations.py
"""

import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# ---------------------------------------------------------------
# 1. Element-wise operations (position by position)
# ---------------------------------------------------------------
print(a + b)          # -> tensor([5., 7., 9.])
print(a - b)          # -> tensor([-3., -3., -3.])
print(a * b)          # -> tensor([ 4., 10., 18.])    NOT matrix multiplication!
print(a / b)          # -> tensor([0.2500, 0.4000, 0.5000])
print(a ** 2)         # -> tensor([1., 4., 9.])
print(a + 10)         # -> tensor([11., 12., 13.])   scalar applied to every element

# Function versions exist too: torch.add(a, b), torch.mul(a, b), ...

# Common math functions
print(torch.sqrt(a))  # -> tensor([1.0000, 1.4142, 1.7321])
print(torch.exp(a))   # -> tensor([ 2.7183,  7.3891, 20.0855])
print(torch.abs(torch.tensor([-2.0, 3.0])))     # -> tensor([2., 3.])

# ---------------------------------------------------------------
# 2. Matrix multiplication
# ---------------------------------------------------------------
# Rule: (n, K) @ (K, m) -> (n, m)   the INNER dimensions must match.
X = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])     # shape (3, 2)  e.g. 3 samples, 2 features
W = torch.tensor([[1.0, 0.0, -1.0],
                  [0.5, 1.0,  2.0]])  # shape (2, 3)  weights
print(X @ W)                       # (3,2) @ (2,3) -> (3,3)
# -> tensor([[2., 2., 3.],
# ->         [5., 4., 5.],
# ->         [8., 6., 7.]])
# Same as torch.matmul(X, W)

# Dot product of two vectors = sum of element-wise products
print(torch.dot(a, b))             # -> tensor(32.)   1*4 + 2*5 + 3*6

# The classic shape error — try uncommenting:
# X @ X   -> RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)
# Fix it by transposing one side: X @ X.T  -> (3,2) @ (2,3) = (3,3)
print((X @ X.T).shape)             # -> torch.Size([3, 3])

# ---------------------------------------------------------------
# 3. Aggregations
# ---------------------------------------------------------------
m = torch.tensor([[1.0, 5.0, 3.0],
                  [4.0, 2.0, 6.0]])
print(m.sum())            # -> tensor(21.)
print(m.mean())           # -> tensor(3.5000)     mean needs a float tensor
print(m.max(), m.min())   # -> tensor(6.) tensor(1.)
print(m.std())            # -> tensor(1.8708)

# dim= chooses the dimension to reduce (like axis= in NumPy)
print(m.sum(dim=0))       # -> tensor([5., 7., 9.])     down the columns
print(m.sum(dim=1))       # -> tensor([ 9., 12.])       across each row

# argmax = INDEX of the largest value. This is how you turn model outputs
# (scores per class) into a predicted class!
scores = torch.tensor([[0.1, 2.5, 0.3],     # sample 0 -> class 1
                       [1.9, 0.2, 0.4]])    # sample 1 -> class 0
print(scores.argmax(dim=1))  # -> tensor([1, 0])

# ---------------------------------------------------------------
# 4. Comparisons produce boolean tensors
# ---------------------------------------------------------------
preds = torch.tensor([1, 0, 2, 2])
labels = torch.tensor([1, 1, 2, 0])
correct = preds == labels
print(correct)                        # -> tensor([ True, False,  True, False])
print(correct.sum().item())           # -> 2
print(correct.float().mean().item())  # -> 0.5     accuracy = 50%

# ---------------------------------------------------------------
# 5. In-place operations (end with underscore _)
# ---------------------------------------------------------------
c = torch.tensor([1.0, 2.0])
c.add_(10)                # modifies c directly, no new tensor
print(c)                  # -> tensor([11., 12.])
# Used inside optimizers; avoid them in your own model code while learning.

# ---------------------------------------------------------------
# 6. Clamping
# ---------------------------------------------------------------
print(torch.tensor([-2.0, 0.5, 3.0]).clamp(min=0))   # -> tensor([0.0000, 0.5000, 3.0000])
# clamp(min=0) is exactly the ReLU activation function!

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Create X of shape (4, 3) and W of shape (3, 2). Compute X @ W and
#    predict its shape before running.
# 2. Given logits = torch.randn(5, 10), get the predicted class for each row.
