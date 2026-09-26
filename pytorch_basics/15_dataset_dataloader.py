"""
PyTorch 15 — Dataset and DataLoader
===================================

Real datasets are too big to push through a model all at once. We train
on small BATCHES instead. PyTorch splits this into two jobs:

  Dataset    -> knows HOW to get ONE sample: len(ds) and ds[i]
                (uses __len__ and __getitem__ from basics lesson 11!)
  DataLoader -> groups samples into BATCHES, SHUFFLES them each epoch,
                and can load in parallel. It's an iterable (basics lesson 16).

Vocabulary:
  batch size -> samples per step (e.g. 32)
  iteration  -> one batch processed (one optimizer.step())
  epoch      -> one full pass over the whole dataset

Run:  python pytorch_basics/15_dataset_dataloader.py
"""

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

torch.manual_seed(0)

# ---------------------------------------------------------------
# 1. A custom Dataset
# ---------------------------------------------------------------
class SquaresDataset(Dataset):
    """Sample i is (x=i, y=i^2)."""

    def __init__(self, n):
        self.x = torch.arange(n, dtype=torch.float32)
        self.y = self.x ** 2

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # In real projects this is where you'd load an image from disk,
        # apply transforms, etc. Return (input, target).
        return self.x[idx], self.y[idx]

ds = SquaresDataset(10)
print(len(ds))        # -> 10
print(ds[3])          # -> (tensor(3.), tensor(9.))

# ---------------------------------------------------------------
# 2. DataLoader: batching
# ---------------------------------------------------------------
loader = DataLoader(ds, batch_size=4, shuffle=False)
for xb, yb in loader:
    print(xb.tolist(), yb.tolist())
# -> [0.0, 1.0, 2.0, 3.0] [0.0, 1.0, 4.0, 9.0]
# -> [4.0, 5.0, 6.0, 7.0] [16.0, 25.0, 36.0, 49.0]
# -> [8.0, 9.0] [64.0, 81.0]                    <- last batch is smaller
print(len(loader))    # -> 3   number of batches per epoch

# ---------------------------------------------------------------
# 3. Shuffling — different order every epoch (helps training)
# ---------------------------------------------------------------
shuffled = DataLoader(ds, batch_size=5, shuffle=True)
for epoch in range(2):
    print(f"epoch {epoch}:", [xb.int().tolist() for xb, _ in shuffled])
# -> epoch 0: [[3, 5, 0, 6, 1], [2, 4, 9, 7, 8]]
# -> epoch 1: [[2, 4, 9, 8, 7], [5, 6, 1, 0, 3]]
# Use shuffle=True for training, shuffle=False for validation/test.

# drop_last=True throws away an incomplete final batch
print(len(DataLoader(ds, batch_size=4, drop_last=True)))   # -> 2

# ---------------------------------------------------------------
# 4. TensorDataset — shortcut when your data is already in tensors
# ---------------------------------------------------------------
X = torch.randn(100, 3)                   # 100 samples, 3 features
y = torch.randint(0, 2, (100,))           # 100 labels (0 or 1)
tds = TensorDataset(X, y)
print(tds[0][0].shape, tds[0][1])         # -> torch.Size([3]) tensor(0)

# ---------------------------------------------------------------
# 5. Splitting into train / validation
# ---------------------------------------------------------------
train_ds, val_ds = random_split(tds, [80, 20])
print(len(train_ds), len(val_ds))         # -> 80 20

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

# Peek at ONE batch — very useful for checking shapes before training
xb, yb = next(iter(train_loader))
print(xb.shape, yb.shape)                 # -> torch.Size([16, 3]) torch.Size([16])

# ---------------------------------------------------------------
# 6. What a training epoch looks like with a DataLoader
# ---------------------------------------------------------------
# for epoch in range(num_epochs):
#     for xb, yb in train_loader:        # one iteration per batch
#         pred = model(xb)
#         loss = loss_fn(pred, yb)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
# Full working example in lesson 16.
#
# Extra DataLoader options you'll meet:
#   num_workers=4   -> load batches in parallel processes (faster for images)
#   pin_memory=True -> faster CPU -> CUDA copies

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Write a Dataset whose samples are (x, 2*x + 1) for x in 0..49 and load
#    it with batch_size=8. How many batches? (Answer: 7)
# 2. Make __getitem__ also return the index, and print it from the loader.
