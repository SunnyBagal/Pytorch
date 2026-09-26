"""
PyTorch 05 — Indexing and Slicing Tensors
=========================================

Same rules as Python lists and NumPy: start at 0, [start:stop:step],
negative indexes count from the end. For multi-dimensional tensors you
separate dimensions with commas: t[row, col].

Run:  python pytorch_basics/05_indexing_slicing.py
"""

import torch

t = torch.arange(1, 13).reshape(3, 4)
print(t)
# -> tensor([[ 1,  2,  3,  4],
# ->         [ 5,  6,  7,  8],
# ->         [ 9, 10, 11, 12]])

# ---------------------------------------------------------------
# 1. Basic indexing
# ---------------------------------------------------------------
print(t[0])          # -> tensor([1, 2, 3, 4])     first row
print(t[0, 1])       # -> tensor(2)                row 0, column 1
print(t[-1, -1])     # -> tensor(12)               last row, last column
print(t[1, 2].item())  # -> 7                      as a Python number

# ---------------------------------------------------------------
# 2. Slicing with :
# ---------------------------------------------------------------
print(t[:, 0])       # -> tensor([1, 5, 9])        ALL rows, column 0
print(t[1, :])       # -> tensor([5, 6, 7, 8])     row 1, ALL columns
print(t[:2, 1:3])    # first 2 rows, columns 1 and 2
# -> tensor([[2, 3],
# ->         [6, 7]])
print(t[:, ::2])     # every other column
# -> tensor([[ 1,  3],
# ->         [ 5,  7],
# ->         [ 9, 11]])

# ---------------------------------------------------------------
# 3. Indexing a batch of images (N, C, H, W)
# ---------------------------------------------------------------
images = torch.rand(8, 3, 32, 32)
print(images[0].shape)            # -> torch.Size([3, 32, 32])   first image
print(images[:4].shape)           # -> torch.Size([4, 3, 32, 32]) first 4 images
print(images[:, 0].shape)         # -> torch.Size([8, 32, 32])    red channel of all images
print(images[0, :, :16, :16].shape)  # -> torch.Size([3, 16, 16]) top-left crop

# ... (Ellipsis) means "all the remaining dimensions"
print(images[..., 0].shape)       # -> torch.Size([8, 3, 32])     last column of pixels

# ---------------------------------------------------------------
# 4. Boolean masks
# ---------------------------------------------------------------
x = torch.tensor([3, -1, 4, -1, 5, -9])
mask = x > 0
print(mask)          # -> tensor([ True, False,  True, False,  True, False])
print(x[mask])       # -> tensor([3, 4, 5])      keep only positives

y = x.clone()        # copy so we don't change x
y[y < 0] = 0         # set all negatives to 0 (this is ReLU again!)
print(y)             # -> tensor([3, 0, 4, 0, 5, 0])

print(torch.where(x > 0, x, torch.zeros_like(x)))   # -> tensor([3, 0, 4, 0, 5, 0])
# torch.where(condition, value_if_true, value_if_false)

# ---------------------------------------------------------------
# 5. Fancy indexing with a list/tensor of indices
# ---------------------------------------------------------------
data = torch.tensor([10, 20, 30, 40, 50])
idx = torch.tensor([4, 0, 2])
print(data[idx])     # -> tensor([50, 10, 30])
# This is how shuffling works: data[torch.randperm(len(data))]

# ---------------------------------------------------------------
# 6. WARNING: slices share memory with the original
# ---------------------------------------------------------------
row = t[0]           # a "view" of t, not a copy
row[0] = 100
print(t[0])          # -> tensor([100,   2,   3,   4])   t changed too!
safe = t[1].clone()  # .clone() makes an independent copy
safe[0] = -1
print(t[1])          # -> tensor([5, 6, 7, 8])           t unchanged

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. From t, get the value 11 and the column [3, 7, 11].
# 2. From torch.arange(20), select all numbers divisible by 3 using a mask.
