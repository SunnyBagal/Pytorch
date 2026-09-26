"""
Lesson 18 — NumPy Basics (the bridge to PyTorch)
================================================

NumPy gives Python a fast N-dimensional ARRAY (ndarray) for math.
PyTorch tensors are designed to feel almost identical to NumPy arrays,
so once you know NumPy, PyTorch tensors will feel familiar.

Why not just lists? Arrays are:
  * FAST   — operations run in optimized C code, not Python loops
  * VECTORIZED — `a * 2` multiplies every element, no loop needed
  * TYPED  — every element has the same dtype (e.g. float32)

Run:  python basics/18_numpy_basics.py
"""

import numpy as np

# ---------------------------------------------------------------
# 1. Creating arrays
# ---------------------------------------------------------------
a = np.array([1, 2, 3])
print(a, a.dtype)             # -> [1 2 3] int64

print(np.zeros((2, 3)))       # 2 rows x 3 cols of zeros
# -> [[0. 0. 0.]
# ->  [0. 0. 0.]]
print(np.ones(3))             # -> [1. 1. 1.]
print(np.arange(0, 10, 2))    # -> [0 2 4 6 8]         like range()
print(np.linspace(0, 1, 5))   # -> [0.   0.25 0.5  0.75 1.  ]  5 evenly spaced points

# ---------------------------------------------------------------
# 2. Shape, ndim, size
# ---------------------------------------------------------------
m = np.array([[1, 2, 3],
              [4, 5, 6]])
print(m.shape)   # -> (2, 3)   2 rows, 3 columns
print(m.ndim)    # -> 2        number of dimensions
print(m.size)    # -> 6        total elements

# ---------------------------------------------------------------
# 3. Vectorized math (no loops!)
# ---------------------------------------------------------------
x = np.array([1.0, 2.0, 3.0])
print(x * 2)            # -> [2. 4. 6.]
print(x + x)            # -> [2. 4. 6.]
print(x ** 2)           # -> [1. 4. 9.]
print(np.sqrt(x))       # -> [1.         1.41421356 1.73205081]
print(x.sum(), x.mean(), x.max())   # -> 6.0 2.0 3.0

# Aggregating along an AXIS: axis=0 goes down rows, axis=1 across columns
print(m.sum(axis=0))    # -> [5 7 9]    column sums
print(m.sum(axis=1))    # -> [ 6 15]    row sums

# ---------------------------------------------------------------
# 4. Indexing, slicing and boolean masks
# ---------------------------------------------------------------
print(m[0, 1])          # -> 2        row 0, col 1
print(m[:, 0])          # -> [1 4]    all rows, column 0
print(m[1, :])          # -> [4 5 6]  row 1, all columns
print(m[m > 3])         # -> [4 5 6]  keep elements matching a condition

# ---------------------------------------------------------------
# 5. Reshaping
# ---------------------------------------------------------------
r = np.arange(6)
print(r.reshape(2, 3))
# -> [[0 1 2]
# ->  [3 4 5]]
print(r.reshape(3, -1).shape)   # -> (3, 2)   -1 = "figure this dimension out"
print(m.T)                      # transpose: rows <-> columns
# -> [[1 4]
# ->  [2 5]
# ->  [3 6]]

# ---------------------------------------------------------------
# 6. Broadcasting — combining arrays of different shapes
# ---------------------------------------------------------------
# NumPy "stretches" smaller arrays to match bigger ones when possible.
row = np.array([10, 20, 30])       # shape (3,)
print(m + row)                     # adds row to EACH row of m
# -> [[11 22 33]
# ->  [14 25 36]]

# ---------------------------------------------------------------
# 7. Matrix multiplication — the core operation of neural networks
# ---------------------------------------------------------------
W = np.array([[1, 0],
              [0, 1],
              [1, 1]])            # shape (3, 2)
print(m @ W)                      # (2,3) @ (3,2) -> (2,2)
# -> [[ 4  5]
# ->  [10 11]]
# Rule: inner dimensions must match: (a, B) @ (B, c) -> (a, c)

# ---------------------------------------------------------------
# 8. Random numbers
# ---------------------------------------------------------------
rng = np.random.default_rng(seed=0)
print(rng.integers(0, 10, size=5))   # -> [8 6 5 2 3]
print(rng.normal(size=2).round(3))   # -> [ 0.105 -0.536]   (mean 0, std 1)

# ---------------------------------------------------------------
# 9. Speed comparison: Python loop vs NumPy
# ---------------------------------------------------------------
import time
data = list(range(1_000_000))
arr = np.arange(1_000_000)

t = time.perf_counter(); _ = [v * 2 for v in data]; py_t = time.perf_counter() - t
t = time.perf_counter(); _ = arr * 2;                np_t = time.perf_counter() - t
print(f"NumPy is ~{py_t / np_t:.0f}x faster here")   # -> NumPy is ~3x faster here (varies a lot by machine; often 3-50x)

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Create a 4x4 array of numbers 0..15 and print its diagonal (np.diag).
# 2. Normalize an array to mean 0 and std 1: (x - x.mean()) / x.std().
# 3. Multiply a (5,3) random matrix by a (3,1) matrix. What's the result shape?
#
# Next up: pytorch_basics/ — tensors are NumPy arrays that can run on a GPU
# and compute gradients automatically.
