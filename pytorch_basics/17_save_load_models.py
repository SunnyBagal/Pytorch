"""
PyTorch 17 — Saving and Loading Models
======================================

Training can take hours — save your work!

The recommended way: save the model's STATE_DICT, a dictionary mapping
each layer's name -> its weight tensor (basics lesson 05: dictionaries).

  torch.save(model.state_dict(), "model.pt")          # save weights
  model = MyModel()                                   # recreate the architecture
  model.load_state_dict(torch.load("model.pt"))       # load the weights in

For resuming training later, save a CHECKPOINT: model + optimizer state +
epoch number, all in one dict.

Run:  python pytorch_basics/17_save_load_models.py
"""

from pathlib import Path

import torch
from torch import nn

torch.manual_seed(0)
out_dir = Path(__file__).parent
weights_path = out_dir / "tiny_model.pt"
ckpt_path = out_dir / "checkpoint.pt"


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


model = TinyNet()

# ---------------------------------------------------------------
# 1. What is a state_dict?
# ---------------------------------------------------------------
sd = model.state_dict()
for name, tensor in sd.items():
    print(f"{name:10s} {tuple(tensor.shape)}")
# -> fc1.weight (8, 4)
# -> fc1.bias   (8,)
# -> fc2.weight (2, 8)
# -> fc2.bias   (2,)

# ---------------------------------------------------------------
# 2. Save and load weights
# ---------------------------------------------------------------
torch.save(model.state_dict(), weights_path)
print(weights_path.name, "saved:", weights_path.exists())   # -> tiny_model.pt saved: True

loaded = TinyNet()                              # fresh model = different random weights
x = torch.rand(1, 4)
print(torch.allclose(model(x), loaded(x)))      # -> False  (not loaded yet)

loaded.load_state_dict(torch.load(weights_path, weights_only=True))
loaded.eval()                                   # set eval mode before inference!
print(torch.allclose(model(x), loaded(x)))      # -> True   identical outputs

# weights_only=True only loads tensors (safer: a .pt file from the internet
# could otherwise run arbitrary code when loaded).

# ---------------------------------------------------------------
# 3. Checkpoints — to RESUME training later
# ---------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
checkpoint = {
    "epoch": 5,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "val_loss": 0.123,
}
torch.save(checkpoint, ckpt_path)

ckpt = torch.load(ckpt_path, weights_only=True)
model2 = TinyNet()
model2.load_state_dict(ckpt["model_state"])
opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
opt2.load_state_dict(ckpt["optimizer_state"])
start_epoch = ckpt["epoch"] + 1
print(f"resuming from epoch {start_epoch}, best val loss {ckpt['val_loss']}")
# -> resuming from epoch 6, best val loss 0.123

# ---------------------------------------------------------------
# 4. Loading onto a different device
# ---------------------------------------------------------------
# A model saved on a GPU can be loaded on a CPU-only laptop with map_location:
cpu_sd = torch.load(weights_path, map_location="cpu", weights_only=True)
print(next(iter(cpu_sd.values())).device)      # -> cpu

# ---------------------------------------------------------------
# 5. Common error: architecture mismatch
# ---------------------------------------------------------------
class DifferentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)             # 16 instead of 8!
        self.fc2 = nn.Linear(16, 2)

try:
    DifferentNet().load_state_dict(torch.load(weights_path, weights_only=True))
except RuntimeError as e:
    print(str(e).splitlines()[0])
    # -> Error(s) in loading state_dict for DifferentNet:
# The class you load into must have the SAME layer names and shapes.

# Clean up demo files
weights_path.unlink()
ckpt_path.unlink()

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. In lesson 16, save the model whenever validation loss improves
#    ("best model" checkpointing), then load it and re-run evaluate().
# 2. Print the size of the saved file with weights_path.stat().st_size.
#
# 🎉 You've finished the basics! Suggested next steps:
#   * torchvision: train a CNN on MNIST / FashionMNIST (nn.Conv2d, transforms)
#   * Learn about overfitting: dropout, weight decay, data augmentation
#   * Transfer learning with a pretrained ResNet
