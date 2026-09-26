"""
Lesson 04 — Tuples and Sets
===========================

TUPLE: like a list but IMMUTABLE (can't change after creation). Written with ().
       Used for fixed groups of values, e.g. an image shape (3, 224, 224).
       In PyTorch, tensor.shape behaves like a tuple!

SET:   an UNORDERED collection of UNIQUE items. Written with {}.
       Great for removing duplicates and fast "is x in here?" checks.

Run:  python basics/04_tuples_and_sets.py
"""

# ---------------------------------------------------------------
# 1. Tuples
# ---------------------------------------------------------------
shape = (3, 224, 224)       # channels, height, width of an image
print(shape[0])             # -> 3
print(len(shape))           # -> 3

# shape[0] = 1              # ERROR: 'tuple' object does not support item assignment

# Tuple UNPACKING: assign each item to its own variable
channels, height, width = shape
print(channels, height, width)   # -> 3 224 224

# Functions often return tuples:
def min_max(values):
    return min(values), max(values)   # the comma makes a tuple

low, high = min_max([4, 8, 1, 9])
print(low, high)                 # -> 1 9

# Swap two variables (Python trick that uses tuples)
x, y = 1, 2
x, y = y, x
print(x, y)                      # -> 2 1

# One-item tuple needs a trailing comma
single = (5,)
print(type(single), type((5)))   # -> <class 'tuple'> <class 'int'>

# ---------------------------------------------------------------
# 2. Sets
# ---------------------------------------------------------------
labels = ["cat", "dog", "cat", "bird", "dog", "cat"]
unique = set(labels)                    # removes duplicates
print(sorted(unique))                   # -> ['bird', 'cat', 'dog']  (sorted for stable order)
print(len(unique))                      # -> 3   number of classes!

print("dog" in unique)                  # -> True   (very fast lookup)

unique.add("fish")
unique.discard("bird")                  # remove if present (no error if missing)
print(sorted(unique))                   # -> ['cat', 'dog', 'fish']

# Set math
a = {1, 2, 3, 4}
b = {3, 4, 5}
print(a | b)    # -> {1, 2, 3, 4, 5}   union        (in either)
print(a & b)    # -> {3, 4}            intersection (in both)
print(a - b)    # -> {1, 2}            difference   (in a, not in b)

# Note: {} creates an empty DICT, not a set. Use set() for an empty set.
print(type({}), type(set()))   # -> <class 'dict'> <class 'set'>

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. You have train_ids = {1,2,3,4,5} and test_ids = {4,5,6}.
#    Find the IDs that leaked into both sets (data leakage check!).
# 2. Unpack the tuple (32, 3, 28, 28) into batch, c, h, w and print them.
