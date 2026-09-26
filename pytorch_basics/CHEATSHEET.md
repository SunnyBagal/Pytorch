# PyTorch Cheatsheet

## Imports
```python
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
```

## Tensors
| Want | Code |
|------|------|
| From data | `torch.tensor([[1, 2], [3, 4]])` |
| Zeros / ones | `torch.zeros(2, 3)`, `torch.ones(2, 3)` |
| Random uniform [0,1) / normal | `torch.rand(2, 3)`, `torch.randn(2, 3)` |
| Range | `torch.arange(0, 10, 2)`, `torch.linspace(0, 1, 5)` |
| Reproducible | `torch.manual_seed(42)` |
| Inspect | `t.shape`, `t.dtype`, `t.device`, `t.ndim`, `t.numel()` |
| Python number | `t.item()`, list: `t.tolist()` |
| Change dtype | `t.float()`, `t.long()`, `t.to(torch.float16)` |
| Copy | `t.clone()` |

## Shapes
| Want | Code |
|------|------|
| New shape | `t.reshape(2, -1)` |
| Flatten all but batch | `t.flatten(start_dim=1)` |
| Add / remove size-1 dim | `t.unsqueeze(0)`, `t.squeeze()` |
| Reorder dims | `t.permute(2, 0, 1)`, 2-D: `t.T` |
| Join existing dim / new dim | `torch.cat([a, b], dim=0)`, `torch.stack([a, b])` |

## Math
| Want | Code |
|------|------|
| Element-wise | `a + b`, `a * b`, `a ** 2`, `torch.exp(a)` |
| Matrix multiply | `a @ b`  — `(n, k) @ (k, m) -> (n, m)` |
| Reduce | `t.sum(dim=1)`, `t.mean()`, `t.max()` |
| Predicted class | `logits.argmax(dim=1)` |
| Accuracy | `(preds == y).float().mean()` |

## Device
```python
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device); x = x.to(device)
arr = t.detach().cpu().numpy()
```

## Autograd
```python
w = torch.tensor(1.0, requires_grad=True)
loss = (w * 3 - 6) ** 2
loss.backward()      # w.grad now holds d(loss)/dw
with torch.no_grad(): ...   # turn tracking off for evaluation
```

## Which loss?
| Task | Last layer outputs | Loss | Target dtype |
|------|-------------------|------|--------------|
| Regression | 1 number | `nn.MSELoss()` | float |
| Binary | 1 logit | `nn.BCEWithLogitsLoss()` | float 0./1. |
| Multi-class | K logits | `nn.CrossEntropyLoss()` | long class index |

## Save / load
```python
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt", weights_only=True)); model.eval()
```

## Debugging the 3 classic errors
| Error message contains | Fix |
|------|-----|
| `shapes cannot be multiplied` | print shapes; check `in_features`; transpose / reshape |
| `expected scalar type Float but found Double/Long` | `.float()` inputs, `.long()` class targets |
| `Expected all tensors to be on the same device` | `.to(device)` on model AND data |
