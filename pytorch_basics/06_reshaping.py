"""
PyTorch 06 — Reshaping: view, reshape, squeeze, unsqueeze, permute, cat, stack
==============================================================================

Most "shape mismatch" bugs are fixed with one of these. Shapes change a lot
as data flows through a network, e.g. flattening an image before a
Linear layer: (N, 1, 28, 28) -> (N, 784).

Run:  python pytorch_basics/06_reshaping.py
"""

import torch

x = torch.arange(12)
print(x)            # -> tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

# ---------------------------------------------------------------
# 1. reshape / view — same data, new shape (total elements must match)
# ---------------------------------------------------------------
print(x.reshape(3, 4))
# -> tensor([[ 0,  1,  2,  3],
# ->         [ 4,  5,  6,  7],
# ->         [ 8,  9, 10, 11]])
print(x.reshape(2, -1).shape)   # -> torch.Size([2, 6])   -1 = "compute this for me"
print(x.view(4, 3).shape)       # -> torch.Size([4, 3])
# view() needs the tensor to be contiguous in memory; reshape() always works.
# When in doubt, use reshape().
# x.reshape(5, 3) -> RuntimeError: shape '[5, 3]' is invalid for input of size 12

# ---------------------------------------------------------------
# 2. flatten — the classic step before a Linear layer
# ---------------------------------------------------------------
batch = torch.rand(32, 1, 28, 28)            # 32 grayscale 28x28 images
flat = batch.flatten(start_dim=1)            # keep the batch dim, flatten the rest
print(flat.shape)                            # -> torch.Size([32, 784])
print(batch.reshape(32, -1).shape)           # -> torch.Size([32, 784])  same result

# ---------------------------------------------------------------
# 3. unsqueeze / squeeze — add or remove dimensions of size 1
# ---------------------------------------------------------------
img = torch.rand(3, 28, 28)                  # one image, no batch dimension
print(img.unsqueeze(0).shape)                # -> torch.Size([1, 3, 28, 28])
# Models expect a batch dimension, so you unsqueeze(0) to predict ONE image.

v = torch.tensor([1, 2, 3])
print(v.unsqueeze(1))                        # column vector
# -> tensor([[1],
# ->         [2],
# ->         [3]])

s = torch.zeros(1, 5, 1)
print(s.squeeze().shape)                     # -> torch.Size([5])     removes ALL size-1 dims
print(s.squeeze(0).shape)                    # -> torch.Size([5, 1])  only dim 0

# ---------------------------------------------------------------
# 4. Transpose and permute — reorder dimensions
# ---------------------------------------------------------------
m = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(m.T)
# -> tensor([[1, 4],
# ->         [2, 5],
# ->         [3, 6]])

# Images from files are usually (H, W, C); PyTorch wants (C, H, W).
hwc = torch.rand(224, 224, 3)
chw = hwc.permute(2, 0, 1)                   # new order: old dims 2, 0, 1
print(chw.shape)                             # -> torch.Size([3, 224, 224])

# ---------------------------------------------------------------
# 5. cat vs stack — combining tensors
# ---------------------------------------------------------------
a = torch.tensor([[1, 2]])
b = torch.tensor([[3, 4]])
print(torch.cat([a, b], dim=0))    # join along an EXISTING dim
# -> tensor([[1, 2],
# ->         [3, 4]])
print(torch.cat([a, b], dim=1))    # -> tensor([[1, 2, 3, 4]])

p = torch.tensor([1, 2])
q = torch.tensor([3, 4])
print(torch.stack([p, q]))         # create a NEW dim (this is how batches are built)
# -> tensor([[1, 2],
# ->         [3, 4]])
print(torch.stack([p, q]).shape)   # -> torch.Size([2, 2])

# ---------------------------------------------------------------
# 6. Splitting
# ---------------------------------------------------------------
parts = torch.arange(10).chunk(3)            # split into 3 roughly equal pieces
print([c.tolist() for c in parts])           # -> [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Turn torch.arange(24) into shape (2, 3, 4), then permute it to (4, 2, 3).
# 2. Stack three tensors of shape (5,) into (3, 5), then into (5, 3) with dim=1.
