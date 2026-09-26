"""
PyTorch 12 — Loss Functions
===========================

A LOSS FUNCTION gives one number that says how wrong the model is.
Training = making this number smaller. Pick the loss that fits the task:

  Task                              Loss                    Model output
  -------------------------------   ---------------------   -------------------------
  Regression (predict a number)     nn.MSELoss / nn.L1Loss  raw number
  Binary classification (yes/no)    nn.BCEWithLogitsLoss    1 raw score (logit)
  Multi-class (one of K classes)    nn.CrossEntropyLoss     K raw scores (logits)

"LOGITS" = raw, unnormalized scores straight from the last Linear layer.
Don't apply softmax/sigmoid yourself before these losses — they do it
internally in a numerically safer way.

Run:  python pytorch_basics/12_loss_functions.py
"""

import torch
from torch import nn

# ---------------------------------------------------------------
# 1. Regression: MSE and L1
# ---------------------------------------------------------------
pred = torch.tensor([2.5, 0.0, 2.0])
target = torch.tensor([3.0, -0.5, 2.0])

mse = nn.MSELoss()        # mean of (pred - target)^2 -> punishes big errors a lot
l1 = nn.L1Loss()          # mean of |pred - target|   -> more robust to outliers
print(mse(pred, target))  # -> tensor(0.1667)    (0.25 + 0.25 + 0) / 3
print(l1(pred, target))   # -> tensor(0.3333)    (0.5 + 0.5 + 0) / 3

# ---------------------------------------------------------------
# 2. Multi-class classification: CrossEntropyLoss
# ---------------------------------------------------------------
# 2 samples, 3 classes. Targets are class INDEXES (dtype long), not one-hot.
logits = torch.tensor([[2.0, 0.5, -1.0],     # model strongly favors class 0
                       [0.1, 0.2,  3.0]])    # model strongly favors class 2
targets = torch.tensor([0, 2])               # correct answers
ce = nn.CrossEntropyLoss()
print(ce(logits, targets))                   # -> tensor(0.1755)   low: predictions are right

wrong_targets = torch.tensor([1, 0])
print(ce(logits, wrong_targets))             # -> tensor(2.3755)   high: predictions are wrong

# Softmax turns logits into probabilities that sum to 1 (for interpretation):
probs = torch.softmax(logits, dim=1)
print(probs)
# -> tensor([[0.7856, 0.1753, 0.0391],
# ->         [0.0493, 0.0545, 0.8962]])
print(probs.sum(dim=1))                      # -> tensor([1., 1.])
print(probs.argmax(dim=1))                   # -> tensor([0, 2])   predicted classes

# ---------------------------------------------------------------
# 3. Binary classification: BCEWithLogitsLoss
# ---------------------------------------------------------------
logit = torch.tensor([3.0, -2.0, 0.5])       # one score per sample
label = torch.tensor([1.0, 0.0, 0.0])        # targets are FLOATS 0.0 / 1.0
bce = nn.BCEWithLogitsLoss()
print(bce(logit, label))                     # -> tensor(0.3832)
print(torch.sigmoid(logit))                  # -> tensor([0.9526, 0.1192, 0.6225])  P(class 1)
print((torch.sigmoid(logit) > 0.5).long())   # -> tensor([1, 0, 1])  predictions

# ---------------------------------------------------------------
# 4. Loss is differentiable -> it's what we call .backward() on
# ---------------------------------------------------------------
w = torch.tensor([1.0, 1.0], requires_grad=True)
loss = mse(w * 2, torch.tensor([4.0, 0.0]))
loss.backward()
print(loss.item(), w.grad)                   # -> 4.0 tensor([-4.,  4.])

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Change logits so that sample 0 favors class 1. Does the loss for
#    targets [0, 2] go up?
# 2. Common bug: pass targets as float to CrossEntropyLoss with shape (N,).
#    Try it and read the error.
