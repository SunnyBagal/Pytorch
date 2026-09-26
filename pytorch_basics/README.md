# Part 2 — PyTorch Basics

Finish [`basics/`](../basics/) first (especially classes, inheritance and NumPy).

| # | Lesson | Key functions |
|---|--------|---------------|
| 01 | [What is a tensor](01_what_is_a_tensor.py) | `torch.tensor`, `.ndim`, `.shape`, `.item()` |
| 02 | [Creating tensors](02_creating_tensors.py) | `zeros`, `ones`, `arange`, `rand`, `randn`, `manual_seed` |
| 03 | [Attributes](03_tensor_attributes.py) | `.dtype`, `.device`, `.float()`, `.to()` |
| 04 | [Operations](04_tensor_operations.py) | `+ - * /`, `@`, `sum(dim=)`, `argmax` |
| 05 | [Indexing & slicing](05_indexing_slicing.py) | `t[:, 0]`, masks, `torch.where`, `.clone()` |
| 06 | [Reshaping](06_reshaping.py) | `reshape`, `flatten`, `unsqueeze`, `permute`, `cat`, `stack` |
| 07 | [NumPy bridge](07_numpy_bridge.py) | `from_numpy`, `.numpy()`, `.detach().cpu()` |
| 08 | [Devices / GPU](08_devices_gpu.py) | `cuda`, `mps`, `.to(device)` |
| 09 | [Autograd](09_autograd.py) | `requires_grad`, `.backward()`, `.grad`, `no_grad` |
| 10 | [Gradient descent by hand](10_gradient_descent_manual.py) | the training loop from scratch |
| 11 | [nn.Module](11_nn_module.py) | `nn.Linear`, `nn.ReLU`, `nn.Sequential`, `parameters()` |
| 12 | [Loss functions](12_loss_functions.py) | `MSELoss`, `CrossEntropyLoss`, `BCEWithLogitsLoss` |
| 13 | [Optimizers](13_optimizers.py) | `SGD`, `Adam`, `zero_grad`, `step`, schedulers |
| 14 | [Project: linear regression](14_linear_regression.py) | full workflow + train/test split |
| 15 | [Dataset & DataLoader](15_dataset_dataloader.py) | `Dataset`, `DataLoader`, `TensorDataset`, `random_split` |
| 16 | [Project: classifier](16_classification_nn.py) | reusable `train_one_epoch` / `evaluate` |
| 17 | [Save & load](17_save_load_models.py) | `state_dict`, `torch.save`, `torch.load`, checkpoints |

See [CHEATSHEET.md](CHEATSHEET.md) for a one-page summary.

## The training loop (memorize this)

```python
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)              # 1. forward
        loss = loss_fn(pred, yb)      # 2. loss
        optimizer.zero_grad()         # 3. clear old gradients
        loss.backward()               # 4. compute gradients
        optimizer.step()              # 5. update weights

    model.eval()
    with torch.no_grad():
        ...                           # measure validation loss / accuracy
```
