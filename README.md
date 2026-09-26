# Learning PyTorch — From Zero

A beginner-friendly, step-by-step path: first the **Python you actually need**, then the **fundamentals of PyTorch**.

Every lesson is a single runnable `.py` file. Each file:

- explains *what* a feature is and *why* you'd use it (in comments),
- shows small examples,
- shows the **output** right next to the code (`# -> ...` or `# Output:`),
- ends with a short **practice** exercise.

## Setup (do this once)

```bash
# 1. Create an isolated Python environment (keeps packages separate per project)
python3 -m venv .venv

# 2. Activate it (you'll see "(.venv)" in your terminal prompt)
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install the libraries
pip install -r requirements.txt
```

## How to study

1. Open a lesson file and read it top to bottom.
2. Run it: `python basics/01_variables_and_types.py`
3. Compare what you see in the terminal with the `# ->` comments.
4. **Change things** and re-run. Breaking code on purpose is the fastest way to learn.
5. Do the practice exercise at the end before moving on.

## Roadmap

| Part | Folder | What you learn |
|------|--------|----------------|
| 1 | [`basics/`](basics/) | Core Python: types, collections, loops, functions, classes, errors, files, generators, decorators, NumPy |
| 2 | [`pytorch_basics/`](pytorch_basics/) | Tensors, operations, GPU, autograd, `nn.Module`, losses, optimizers, training loops, datasets, saving models |

Take your time with Part 1 — PyTorch is "just Python", so strong Python makes PyTorch easy.

Quick reference: [`pytorch_basics/CHEATSHEET.md`](pytorch_basics/CHEATSHEET.md)

## My progress

Tick these off as you go (edit this file and change `[ ]` to `[x]`):

- [ ] Python basics 01–06 (types, strings, collections, conditionals)
- [ ] Python basics 07–12 (loops, functions, classes, inheritance)
- [ ] Python basics 13–18 (errors, modules, files, generators, decorators, NumPy)
- [ ] PyTorch 01–08 (tensors, operations, shapes, devices)
- [ ] PyTorch 09–13 (autograd, nn.Module, losses, optimizers)
- [ ] PyTorch 14–17 (full projects, DataLoader, saving models)
