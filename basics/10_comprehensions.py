"""
Lesson 10 — Comprehensions
==========================

A COMPREHENSION builds a list/dict/set in ONE readable line.

    [expression  for item in iterable  if condition]

It replaces the "create empty list, loop, append" pattern.

Run:  python basics/10_comprehensions.py
"""

# ---------------------------------------------------------------
# 1. List comprehension vs normal loop
# ---------------------------------------------------------------
squares = []
for x in range(5):
    squares.append(x ** 2)
print(squares)                                # -> [0, 1, 4, 9, 16]

squares = [x ** 2 for x in range(5)]          # same thing, one line
print(squares)                                # -> [0, 1, 4, 9, 16]

# ---------------------------------------------------------------
# 2. With a condition (filter)
# ---------------------------------------------------------------
evens = [x for x in range(10) if x % 2 == 0]
print(evens)                                  # -> [0, 2, 4, 6, 8]

# With if/else (transform) — note the if/else goes BEFORE `for`
labels = ["pos" if p >= 0.5 else "neg" for p in [0.9, 0.2, 0.6]]
print(labels)                                 # -> ['pos', 'neg', 'pos']

# ---------------------------------------------------------------
# 3. Dict and set comprehensions
# ---------------------------------------------------------------
words = ["cat", "horse", "ox"]
lengths = {w: len(w) for w in words}
print(lengths)                                # -> {'cat': 3, 'horse': 5, 'ox': 2}

first_letters = {w[0] for w in ["apple", "avocado", "banana"]}
print(sorted(first_letters))                  # -> ['a', 'b']

# ---------------------------------------------------------------
# 4. Nested comprehension (e.g. flatten or build a matrix)
# ---------------------------------------------------------------
matrix = [[1, 2], [3, 4], [5, 6]]
flat = [v for row in matrix for v in row]
print(flat)                                   # -> [1, 2, 3, 4, 5, 6]

identity = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
print(identity)                               # -> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# Transpose (rows <-> columns) — tensors will do this with .T
transposed = [[row[i] for row in matrix] for i in range(2)]
print(transposed)                             # -> [[1, 3, 5], [2, 4, 6]]

# ---------------------------------------------------------------
# 5. Generator expression — like a list comp but with () and lazy
# ---------------------------------------------------------------
# Doesn't build the whole list in memory; computes values on demand.
total = sum(x * x for x in range(1_000_000))
print(total)                                  # -> 333332833333500000

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. From names = ["img1.png", "notes.txt", "img2.png"], keep only .png files.
# 2. Given preds=[1,0,1,1] and targets=[1,1,1,0], compute accuracy using
#    sum(p == t for p, t in zip(preds, targets)) / len(preds).
