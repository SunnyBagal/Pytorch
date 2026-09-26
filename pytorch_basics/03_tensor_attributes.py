"""
PyTorch 03 — Tensor Attributes: shape, dtype, device
====================================================

The THREE most common PyTorch errors come from mismatches in:

  1. shape  — "mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)"
  2. dtype  — "expected scalar type Float but found Long"
  3. device — "Expected all tensors to be on the same device, cpu and cuda:0"

When stuck, print these three things!

Run:  python pytorch_basics/03_tensor_attributes.py
"""

import torch

x = torch.rand(3, 4)

print(x.shape)       # -> torch.Size([3, 4])
print(x.size())      # -> torch.Size([3, 4])    same thing as .shape
print(x.size(1))     # -> 4                     size of dimension 1
print(x.dtype)       # -> torch.float32         the data type
print(x.device)      # -> cpu                   where it lives

# ---------------------------------------------------------------
# 1. dtypes (data types)
# ---------------------------------------------------------------
# float32 -> default for weights, inputs, most math (good precision/speed balance)
# float16 / bfloat16 -> half precision, faster on modern GPUs, less precise
# float64 -> double precision (rarely needed in deep learning)
# int64 (long) -> default for integers; class LABELS for classification
# bool -> True/False masks
print(torch.tensor([1, 2]).dtype)       # -> torch.int64     ints default to int64
print(torch.tensor([1.0, 2]).dtype)     # -> torch.float32   floats default to float32
print(torch.tensor([True]).dtype)       # -> torch.bool

f16 = torch.tensor([1.0, 2.0], dtype=torch.float16)   # choose the dtype on creation
print(f16.dtype)                        # -> torch.float16

# ---------------------------------------------------------------
# 2. Converting dtypes
# ---------------------------------------------------------------
ints = torch.tensor([1, 2, 3])
print(ints.float())                     # -> tensor([1., 2., 3.])
print(ints.to(torch.float64))           # -> tensor([1., 2., 3.], dtype=torch.float64)
print(torch.tensor([1.7, 2.2]).long())  # -> tensor([1, 2])   truncates toward zero

# Mixed dtypes are auto-promoted to the "bigger" type:
print((ints + torch.tensor([0.5, 0.5, 0.5])).dtype)   # -> torch.float32

# ---------------------------------------------------------------
# 3. Memory used
# ---------------------------------------------------------------
print(x.element_size())                 # -> 4    bytes per float32 element
print(x.element_size() * x.numel())     # -> 48   total bytes (12 elements * 4)
# float16 halves memory — that's why big models use it.

# ---------------------------------------------------------------
# 4. A handy debug helper
# ---------------------------------------------------------------
def describe(name, t):
    print(f"{name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")

describe("x", x)          # -> x: shape=(3, 4) dtype=torch.float32 device=cpu
describe("ints", ints)    # -> ints: shape=(3,) dtype=torch.int64 device=cpu

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Create a float64 tensor and convert it to float32. Compare element_size().
# 2. Try adding a float16 and a float32 tensor. What dtype is the result?
