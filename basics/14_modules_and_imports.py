"""
Lesson 14 — Modules, Imports and Useful Standard Libraries
==========================================================

A MODULE is a .py file with code you can reuse. A PACKAGE is a folder of
modules. You bring them into your file with `import`.

Libraries you install with pip (torch, numpy) are packages too.

Run:  python basics/14_modules_and_imports.py
"""

# ---------------------------------------------------------------
# 1. Import styles
# ---------------------------------------------------------------
import math                         # import the whole module
print(math.sqrt(16), math.pi)       # -> 4.0 3.141592653589793

from math import exp, log           # import specific names
print(round(exp(1), 4), log(1))     # -> 2.7183 0.0

import numpy as np                  # import with an ALIAS (nickname)
# The standard aliases in ML code:
#   import numpy as np
#   import torch
#   import torch.nn as nn
#   import torch.nn.functional as F

# ---------------------------------------------------------------
# 2. random — random numbers (and why SEEDS matter)
# ---------------------------------------------------------------
import random

random.seed(42)                     # fix the seed -> same "random" numbers every run
print(random.randint(1, 10))        # -> 2
print(random.choice(["a", "b", "c"]))  # -> a
nums = [1, 2, 3, 4, 5]
random.shuffle(nums)                # shuffles IN PLACE
print(nums)                         # -> [4, 5, 1, 2, 3]
# In PyTorch you use torch.manual_seed(42) for reproducible experiments.

# ---------------------------------------------------------------
# 3. os and pathlib — working with files and folders
# ---------------------------------------------------------------
import os
from pathlib import Path

here = Path(__file__).parent        # folder containing this file
print(here.name)                    # -> basics
print((here / "data" / "img.png").as_posix().endswith("basics/data/img.png"))  # -> True
# The `/` operator joins paths safely on every OS.

print(os.path.basename("/a/b/model.pt"))     # -> model.pt
print(Path("model.pt").suffix)               # -> .pt

# ---------------------------------------------------------------
# 4. time — measure how long something takes
# ---------------------------------------------------------------
import time
start = time.perf_counter()
_ = sum(range(1_000_000))
elapsed = time.perf_counter() - start
print(f"took {elapsed:.4f}s")       # -> took 0.0xxx s (varies by machine)

# ---------------------------------------------------------------
# 5. collections — handy data structures
# ---------------------------------------------------------------
from collections import Counter, defaultdict

labels = ["cat", "dog", "cat", "cat", "bird"]
print(Counter(labels))              # -> Counter({'cat': 3, 'dog': 1, 'bird': 1})
print(Counter(labels).most_common(1))   # -> [('cat', 3)]   class imbalance check!

groups = defaultdict(list)          # missing keys get a default empty list
for name, lbl in [("a.png", "cat"), ("b.png", "dog"), ("c.png", "cat")]:
    groups[lbl].append(name)
print(dict(groups))                 # -> {'cat': ['a.png', 'c.png'], 'dog': ['b.png']}

# ---------------------------------------------------------------
# 6. if __name__ == "__main__":
# ---------------------------------------------------------------
# Code inside this block runs ONLY when you run the file directly,
# NOT when another file imports it. Every training script uses this.
def main():
    print("running as a script")

if __name__ == "__main__":
    main()                          # -> running as a script

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Create my_utils.py next to this file with a function, then import it.
# 2. Use Counter to find the most common letter in "mississippi".
