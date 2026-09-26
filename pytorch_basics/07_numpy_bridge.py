"""
PyTorch 07 — NumPy <-> PyTorch
==============================

Lots of data comes as NumPy arrays (from pandas, OpenCV, scikit-learn...).
Converting between the two is easy — but watch out for SHARED MEMORY and dtypes.

  torch.from_numpy(arr)  -> tensor  (shares memory with arr)
  torch.tensor(arr)      -> tensor  (makes a copy)
  tensor.numpy()         -> array   (shares memory; tensor must be on CPU)

Run:  python pytorch_basics/07_numpy_bridge.py
"""

import numpy as np
import torch

# ---------------------------------------------------------------
# 1. NumPy -> PyTorch
# ---------------------------------------------------------------
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)
print(t)              # -> tensor([1., 2., 3.], dtype=torch.float64)
# NOTE: NumPy's default float is float64, but PyTorch models use float32!
t32 = torch.from_numpy(arr).float()
print(t32.dtype)      # -> torch.float32
# Forgetting this causes: "expected scalar type Float but found Double"

# ---------------------------------------------------------------
# 2. Shared memory
# ---------------------------------------------------------------
arr[0] = 100
print(t)              # -> tensor([100.,   2.,   3.], dtype=torch.float64)  changed too!

copy = torch.tensor(arr)   # independent copy
arr[1] = 200
print(copy)           # -> tensor([100.,   2.,   3.], dtype=torch.float64)  not affected

# ---------------------------------------------------------------
# 3. PyTorch -> NumPy
# ---------------------------------------------------------------
x = torch.ones(3)
n = x.numpy()
print(n, type(n))     # -> [1. 1. 1.] <class 'numpy.ndarray'>

# If the tensor tracks gradients or is on a GPU, you need:
#   tensor.detach().cpu().numpy()
w = torch.ones(2, requires_grad=True)
print(w.detach().cpu().numpy())   # -> [1. 1.]
# (w.numpy() would raise: "Can't call numpy() on Tensor that requires grad")

# ---------------------------------------------------------------
# 4. To plain Python
# ---------------------------------------------------------------
print(torch.tensor([[1, 2], [3, 4]]).tolist())   # -> [[1, 2], [3, 4]]
print(torch.tensor(3.5).item())                   # -> 3.5

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Create np.random.rand(2, 3), convert to a float32 tensor, multiply by
#    2, and convert back to NumPy.
# 2. Prove torch.from_numpy shares memory by changing the tensor and
#    printing the array.
