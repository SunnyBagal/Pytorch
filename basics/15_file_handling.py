"""
Lesson 15 — Reading and Writing Files (text, CSV, JSON)
=======================================================

Datasets, configs, logs and results all live in files.

`with open(...) as f:` is the safe way to open a file: it CLOSES the file
automatically, even if an error happens (this is a "context manager").

Modes: "r" read (default), "w" write (overwrites!), "a" append.

Run:  python basics/15_file_handling.py
"""

import csv
import json
from pathlib import Path

folder = Path(__file__).parent
txt_path = folder / "notes_demo.txt"

# ---------------------------------------------------------------
# 1. Writing and reading text
# ---------------------------------------------------------------
with open(txt_path, "w") as f:
    f.write("epoch 1 loss 0.90\n")      # \n = newline
    f.write("epoch 2 loss 0.55\n")

with open(txt_path, "a") as f:          # append, don't overwrite
    f.write("epoch 3 loss 0.31\n")

with open(txt_path) as f:
    content = f.read()                  # whole file as one string
print(content, end="")
# -> epoch 1 loss 0.90
# -> epoch 2 loss 0.55
# -> epoch 3 loss 0.31

# Read line by line and parse numbers out of it
with open(txt_path) as f:
    losses = [float(line.split()[-1]) for line in f]
print(losses)                           # -> [0.9, 0.55, 0.31]

# ---------------------------------------------------------------
# 2. JSON — perfect for configs and results (maps to dicts/lists)
# ---------------------------------------------------------------
config = {"model": "mlp", "lr": 0.01, "layers": [64, 32]}
json_text = json.dumps(config, indent=2)    # dict -> JSON string
print(json_text)
# -> {
# ->   "model": "mlp",
# ->   "lr": 0.01,
# ->   "layers": [
# ->     64,
# ->     32
# ->   ]
# -> }
loaded = json.loads(json_text)              # JSON string -> dict
print(loaded["layers"][0])                  # -> 64
# To files: json.dump(obj, f) and json.load(f)

# ---------------------------------------------------------------
# 3. CSV — tables of data
# ---------------------------------------------------------------
import io
csv_text = "height,weight,label\n1.7,65,0\n1.8,90,1\n"
reader = csv.DictReader(io.StringIO(csv_text))   # io.StringIO pretends a string is a file
rows = list(reader)
print(rows[0])                 # -> {'height': '1.7', 'weight': '65', 'label': '0'}
# Note: CSV values come in as STRINGS — convert them before doing math:
heights = [float(r["height"]) for r in rows]
print(heights)                 # -> [1.7, 1.8]

# ---------------------------------------------------------------
# 4. Checking and cleaning up files with pathlib
# ---------------------------------------------------------------
print(txt_path.exists())       # -> True
txt_path.unlink()              # delete the demo file
print(txt_path.exists())       # -> False

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Save a dict of hyperparameters to "config.json" with json.dump,
#    then load it back with json.load.
# 2. Write a CSV with csv.writer containing epoch,loss rows.
