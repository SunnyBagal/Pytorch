"""
Lesson 05 — Dictionaries
========================

A DICTIONARY (dict) stores KEY -> VALUE pairs. You look things up by key
instead of by position. Keys must be unique and immutable (str, int, tuple).

In ML you'll see dicts for: hyperparameters/configs, label <-> index
mappings, training history, and model.state_dict() in PyTorch
(which maps layer names -> weight tensors).

Run:  python basics/05_dictionaries.py
"""

# ---------------------------------------------------------------
# 1. Creating and reading
# ---------------------------------------------------------------
config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10,
}
print(config["batch_size"])            # -> 32
print(config.get("dropout"))           # -> None   (.get won't crash if key missing)
print(config.get("dropout", 0.5))      # -> 0.5    (default value)
# config["dropout"]                    # KeyError! key doesn't exist

# ---------------------------------------------------------------
# 2. Adding, changing, removing
# ---------------------------------------------------------------
config["epochs"] = 20                  # update
config["optimizer"] = "adam"           # add new key
removed = config.pop("batch_size")     # remove and return the value
print(removed)                         # -> 32
print(config)
# -> {'learning_rate': 0.001, 'epochs': 20, 'optimizer': 'adam'}

print("epochs" in config)              # -> True   (checks KEYS)

# ---------------------------------------------------------------
# 3. Looping over a dict
# ---------------------------------------------------------------
for key, value in config.items():      # .items() gives (key, value) pairs
    print(f"{key:>14}: {value}")
# ->  learning_rate: 0.001     ({key:>14} right-aligns the key in 14 chars)
# ->        epochs: 20
# ->     optimizer: adam

print(list(config.keys()))    # -> ['learning_rate', 'epochs', 'optimizer']
print(list(config.values()))  # -> [0.001, 20, 'adam']

# ---------------------------------------------------------------
# 4. Real ML example: mapping class names <-> numbers
# ---------------------------------------------------------------
# Neural networks only understand numbers, so we map labels to integers.
classes = ["cat", "dog", "bird"]
class_to_idx = {name: i for i, name in enumerate(classes)}   # dict comprehension
idx_to_class = {i: name for name, i in class_to_idx.items()}
print(class_to_idx)   # -> {'cat': 0, 'dog': 1, 'bird': 2}
print(idx_to_class[1])  # -> dog   (turn the model's prediction back into a name)

# ---------------------------------------------------------------
# 5. Counting things & nested dicts
# ---------------------------------------------------------------
words = ["a", "b", "a", "c", "a", "b"]
counts = {}
for w in words:
    counts[w] = counts.get(w, 0) + 1
print(counts)         # -> {'a': 3, 'b': 2, 'c': 1}

history = {"train": {"loss": [0.9, 0.6]}, "val": {"loss": [1.0, 0.8]}}
print(history["val"]["loss"][-1])   # -> 0.8   latest validation loss

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Create a dict of 3 students -> marks. Print the student with the
#    highest mark (hint: max(d, key=d.get)).
# 2. Build a history dict {"loss": []} and append 3 values to the list.
