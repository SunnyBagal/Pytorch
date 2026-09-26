"""
PyTorch 08 — Devices: CPU, CUDA (NVIDIA GPU) and MPS (Apple GPU)
================================================================

GPUs do thousands of multiplications in parallel, so training on them is
MUCH faster. PyTorch supports:
  * "cpu"  — always available
  * "cuda" — NVIDIA GPUs
  * "mps"  — Apple Silicon GPUs (M1/M2/M3/M4 Macs)

THE GOLDEN RULE: all tensors in one operation (and the model) must be on
the SAME device. Move things with .to(device).

Run:  python pytorch_basics/08_devices_gpu.py
"""

import time
import torch

# ---------------------------------------------------------------
# 1. Pick the best available device (copy this into every project)
# ---------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)    # -> Using device: mps   (on an Apple Silicon Mac)

# ---------------------------------------------------------------
# 2. Creating / moving tensors
# ---------------------------------------------------------------
a = torch.rand(3)                          # created on CPU
b = a.to(device)                           # a COPY on the device
c = torch.zeros(3, device=device)          # create directly on the device
print(a.device, b.device, c.device)        # -> cpu mps:0 mps:0

# Mixing devices fails (only possible to show if you have a GPU):
if device.type != "cpu":
    try:
        a + b                              # a is on CPU, b is on the GPU
    except RuntimeError as e:
        print("Error:", str(e)[:55], "...")
        # -> Error: Expected all tensors to be on the same device, but ...

print((b + c).device)                      # -> mps:0   fine: both on the device

# Back to CPU (needed for .numpy() or printing with other libraries)
print(b.cpu().device)                      # -> cpu

# ---------------------------------------------------------------
# 3. Moving a model (preview — models are covered in lesson 11)
# ---------------------------------------------------------------
model = torch.nn.Linear(4, 2).to(device)   # moves all its weights
x = torch.rand(5, 4, device=device)        # data must go to the same device
print(model(x).device)                     # -> mps:0

# ---------------------------------------------------------------
# 4. Speed test: large matrix multiplication
# ---------------------------------------------------------------
def bench(dev, n=2048, reps=10):
    m = torch.rand(n, n, device=dev)
    _ = m @ m                              # warm-up (first call is slower)
    if dev.type == "mps":
        torch.mps.synchronize()            # GPUs run asynchronously; wait for them
    elif dev.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        _ = m @ m
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start

cpu_t = bench(torch.device("cpu"))
print(f"CPU:  {cpu_t:.3f}s")
if device.type != "cpu":
    dev_t = bench(device)
    print(f"{device.type.upper()}:  {dev_t:.3f}s  (~{cpu_t / dev_t:.1f}x faster)")
# Output varies by machine, e.g.
# -> CPU:  0.098s
# -> MPS:  0.044s  (~2.2x faster)   (an M-series Mac; bigger n = bigger gap)
# For tiny tensors the CPU can actually win: moving data has a cost.

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Try bench() with n=128. Is the GPU still faster?
# 2. Create a tensor on your device and try .numpy() on it. Read the error,
#    then fix it with .cpu().
