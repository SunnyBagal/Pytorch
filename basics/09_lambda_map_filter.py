"""
Lesson 09 — lambda, map, filter, sorted(key=...)
================================================

LAMBDA: a tiny, anonymous (unnamed) one-line function.
        lambda arguments: expression

Handy when a function needs ANOTHER function as input, e.g. for sorting
or in PyTorch's `transforms.Lambda(lambda x: x / 255)`.

Run:  python basics/09_lambda_map_filter.py
"""

# ---------------------------------------------------------------
# 1. lambda
# ---------------------------------------------------------------
square = lambda x: x ** 2          # same as: def square(x): return x ** 2
add = lambda a, b: a + b
print(square(5), add(2, 3))        # -> 25 5

# ---------------------------------------------------------------
# 2. map(function, iterable) — apply a function to every item
# ---------------------------------------------------------------
pixels = [0, 128, 255]
normalized = list(map(lambda p: p / 255, pixels))
print(normalized)                  # -> [0.0, 0.5019607843137255, 1.0]

print(list(map(int, ["1", "2", "3"])))   # -> [1, 2, 3]   works with any function

# ---------------------------------------------------------------
# 3. filter(function, iterable) — keep items where function is True
# ---------------------------------------------------------------
nums = [1, 2, 3, 4, 5, 6]
evens = list(filter(lambda n: n % 2 == 0, nums))
print(evens)                       # -> [2, 4, 6]

# ---------------------------------------------------------------
# 4. sorted / max / min with key=
# ---------------------------------------------------------------
# `key` tells Python WHAT to compare.
models = [("resnet", 0.91), ("vgg", 0.88), ("vit", 0.94)]
by_acc = sorted(models, key=lambda m: m[1], reverse=True)
print(by_acc)                      # -> [('vit', 0.94), ('resnet', 0.91), ('vgg', 0.88)]

best = max(models, key=lambda m: m[1])
print(best)                        # -> ('vit', 0.94)

words = ["banana", "kiwi", "apple"]
print(sorted(words, key=len))      # -> ['kiwi', 'apple', 'banana']

# ---------------------------------------------------------------
# 5. any() and all()
# ---------------------------------------------------------------
losses = [0.5, 0.3, float("nan")]
import math
print(any(math.isnan(l) for l in losses))   # -> True   (is ANY loss NaN?)
print(all(l > 0 for l in [1, 2, 3]))        # -> True   (are ALL positive?)

# Tip: list comprehensions (next lesson) are often clearer than map/filter.

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Use map + lambda to convert Celsius [0, 25, 100] to Fahrenheit (c*9/5+32).
# 2. Sort a list of dicts [{"name": "a", "age": 30}, ...] by age.
