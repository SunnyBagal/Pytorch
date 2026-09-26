"""
PyTorch 02 — Creating Tensors
=============================

You rarely type tensors by hand. Instead you create them with factory
functions: zeros, ones, random values, ranges, or "like another tensor".

Neural network weights start as RANDOM numbers, so random tensors matter.

Run:  python pytorch_basics/02_creating_tensors.py
"""

import torch

# ---------------------------------------------------------------
# 1. From Python data
# ---------------------------------------------------------------
print(torch.tensor([[1, 2], [3, 4]]))
# -> tensor([[1, 2],
# ->         [3, 4]])

# ---------------------------------------------------------------
# 2. Filled tensors
# ---------------------------------------------------------------
print(torch.zeros(2, 3))
# -> tensor([[0., 0., 0.],
# ->         [0., 0., 0.]])
print(torch.ones(2, 2))
# -> tensor([[1., 1.],
# ->         [1., 1.]])
print(torch.full((2, 2), 7.0))        # every element = 7
# -> tensor([[7., 7.],
# ->         [7., 7.]])
print(torch.eye(3))                   # identity matrix (1s on the diagonal)
# -> tensor([[1., 0., 0.],
# ->         [0., 1., 0.],
# ->         [0., 0., 1.]])

# ---------------------------------------------------------------
# 3. Ranges
# ---------------------------------------------------------------
print(torch.arange(0, 10, 2))         # -> tensor([0, 2, 4, 6, 8])    like range()
print(torch.linspace(0, 1, steps=5))  # -> tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

# ---------------------------------------------------------------
# 4. Random tensors (+ seeds for reproducibility)
# ---------------------------------------------------------------
# torch.manual_seed fixes the random number generator so you get the SAME
# "random" numbers every run. Always set it while learning/debugging.
torch.manual_seed(42)

print(torch.rand(2, 3))       # uniform random in [0, 1)
# -> tensor([[0.8823, 0.9150, 0.3829],
# ->         [0.9593, 0.3904, 0.6009]])

print(torch.randn(3))         # "normal" distribution: mean 0, std 1 (can be negative)
# -> tensor([ 1.1561,  0.3965, -2.4661])

print(torch.randint(0, 10, (5,)))   # random integers in [0, 10), shape (5,)
# -> tensor([1, 2, 5, 5, 7])

print(torch.randperm(5))      # a random ordering of 0..4 (used for shuffling)
# -> tensor([1, 2, 4, 0, 3])

# Proof that the seed makes things reproducible:
torch.manual_seed(0); a = torch.rand(3)
torch.manual_seed(0); b = torch.rand(3)
print(torch.equal(a, b))      # -> True

# ---------------------------------------------------------------
# 5. "_like" functions — same shape (and dtype/device) as another tensor
# ---------------------------------------------------------------
x = torch.tensor([[1.0, 2.0, 3.0]])
print(torch.zeros_like(x))    # -> tensor([[0., 0., 0.]])
print(torch.ones_like(x))     # -> tensor([[1., 1., 1.]])
print(torch.rand_like(x).shape)   # -> torch.Size([1, 3])

# ---------------------------------------------------------------
# 6. Uninitialized (fast, but contains garbage values)
# ---------------------------------------------------------------
e = torch.empty(2, 2)         # memory is allocated but NOT cleared
print(e.shape)                # -> torch.Size([2, 2])
# Only use empty() if you will overwrite every value.

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Create a random tensor of shape (3, 4) with a seed of 123. Run the file
#    twice — do you get the same numbers?
# 2. Create a tensor 1, 1.5, 2, ..., 5 using arange with a float step.
