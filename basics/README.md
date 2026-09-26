# Part 1 — Python Basics

Everything in Python you need before PyTorch. Do them in order.

| # | Lesson | Why it matters for PyTorch |
|---|--------|----------------------------|
| 01 | [Variables & types](01_variables_and_types.py) | ints/floats map to tensor dtypes |
| 02 | [Strings](02_strings.py) | slicing rules are the same for tensors; f-strings for training logs |
| 03 | [Lists](03_lists.py) | storing losses; the aliasing trap (tensors behave the same) |
| 04 | [Tuples & sets](04_tuples_and_sets.py) | `tensor.shape` is a tuple; unpacking `images, labels = batch` |
| 05 | [Dictionaries](05_dictionaries.py) | configs, class↔index maps, `state_dict()` |
| 06 | [Conditionals](06_conditionals.py) | `device = "cuda" if ... else "cpu"` |
| 07 | [Loops](07_loops.py) | the training loop, early stopping |
| 08 | [Functions](08_functions.py) | `train_one_epoch()`, keyword args like `nn.Linear(in_features=...)` |
| 09 | [lambda / map / filter](09_lambda_map_filter.py) | quick transforms, sorting results |
| 10 | [Comprehensions](10_comprehensions.py) | concise data processing |
| 11 | [Classes & objects](11_classes_and_objects.py) | **every model is a class**; `__call__`, `__len__`, `__getitem__` |
| 12 | [Inheritance](12_inheritance.py) | `class Net(nn.Module)` and `super().__init__()` |
| 13 | [Exceptions](13_exceptions.py) | reading shape-mismatch errors |
| 14 | [Modules & imports](14_modules_and_imports.py) | `import torch.nn as nn`, seeds, paths |
| 15 | [File handling](15_file_handling.py) | loading data and configs |
| 16 | [Iterators & generators](16_iterators_generators.py) | how DataLoader hands you batches |
| 17 | [Decorators & context managers](17_decorators_and_context_managers.py) | `@torch.no_grad()` and `with torch.no_grad():` |
| 18 | [NumPy basics](18_numpy_basics.py) | tensors are "NumPy on steroids" |

Run any lesson with:

```bash
python basics/11_classes_and_objects.py
```

**Must-know before moving on:** 03, 05, 07, 08, 11, 12, 18.
