"""
PyTorch 01 — What is a Tensor?
==============================

PyTorch is a library for building and training neural networks.
Its core data structure is the TENSOR.

A TENSOR is a multi-dimensional array of numbers — just like a NumPy array —
with two superpowers:
  1. It can live on a GPU (much faster for big math).
  2. It can track operations to compute GRADIENTS automatically (autograd),
     which is how neural networks learn.

Dimensions ("rank") by example:
  0-D  scalar   ->  a single number          e.g. a loss value
  1-D  vector   ->  a list of numbers        e.g. one data sample's features
  2-D  matrix   ->  rows x columns           e.g. a table / batch of samples
  3-D           ->  e.g. a grayscale video, or a color image (C, H, W)
  4-D           ->  e.g. a BATCH of color images (N, C, H, W)

Run:  python pytorch_basics/01_what_is_a_tensor.py
"""

import torch   # the main PyTorch package

print(torch.__version__)            # -> 2.x.x  (your installed version)

# ---------------------------------------------------------------
# 1. Scalar (0-D)
# ---------------------------------------------------------------
scalar = torch.tensor(7)
print(scalar)             # -> tensor(7)
print(scalar.ndim)        # -> 0          number of dimensions
print(scalar.item())      # -> 7          .item() converts a 1-element tensor to a Python number

# ---------------------------------------------------------------
# 2. Vector (1-D)
# ---------------------------------------------------------------
vector = torch.tensor([1.0, 2.0, 3.0])
print(vector)             # -> tensor([1., 2., 3.])
print(vector.ndim)        # -> 1
print(vector.shape)       # -> torch.Size([3])     shape is tuple-like

# ---------------------------------------------------------------
# 3. Matrix (2-D)
# ---------------------------------------------------------------
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
print(matrix)
# -> tensor([[1, 2, 3],
# ->         [4, 5, 6]])
print(matrix.ndim)        # -> 2
print(matrix.shape)       # -> torch.Size([2, 3])   2 rows, 3 columns

# Tip: count the opening square brackets [[ to know the number of dimensions.

# ---------------------------------------------------------------
# 4. 3-D tensor and a batch of images (4-D)
# ---------------------------------------------------------------
t3 = torch.tensor([[[1, 2],
                    [3, 4]],
                   [[5, 6],
                    [7, 8]]])
print(t3.shape)           # -> torch.Size([2, 2, 2])

# A batch of 16 RGB images of size 32x32 pixels:
images = torch.zeros(16, 3, 32, 32)
print(images.shape)       # -> torch.Size([16, 3, 32, 32])
print(images.ndim)        # -> 4
print(images.numel())     # -> 49152   total number of elements (16*3*32*32)

# ---------------------------------------------------------------
# 5. Tensors vs Python lists
# ---------------------------------------------------------------
py_list = [1, 2, 3]
print(py_list * 2)                    # -> [1, 2, 3, 1, 2, 3]   list repeats
print(torch.tensor(py_list) * 2)      # -> tensor([2, 4, 6])    tensor does math!

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Create a tensor representing a 5x5 grayscale image (1 channel).
#    What shape should it have? (hint: (1, 5, 5))
# 2. Create a 2-D tensor of your 3 favorite numbers in each of 2 rows
#    and print ndim, shape and numel().
