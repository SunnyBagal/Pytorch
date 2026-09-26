"""
Lesson 16 — Iterators and Generators
====================================

ITERABLE: anything you can loop over (list, str, dict, range, file...).
ITERATOR: the object that hands out items ONE AT A TIME via next().
GENERATOR: the easiest way to make an iterator — a function with `yield`.

Why care? PyTorch's DataLoader is an iterable: each loop step hands you
the next batch, without loading the whole dataset into memory at once.

Run:  python basics/16_iterators_generators.py
"""

# ---------------------------------------------------------------
# 1. iter() and next()
# ---------------------------------------------------------------
it = iter([10, 20, 30])
print(next(it))      # -> 10
print(next(it))      # -> 20
print(next(it))      # -> 30
# next(it)           # StopIteration: no more items
# A `for` loop simply calls next() until StopIteration.

# You'll see this in PyTorch to grab ONE batch for a quick look:
#   images, labels = next(iter(train_loader))

# ---------------------------------------------------------------
# 2. Generators with yield
# ---------------------------------------------------------------
def count_up(n):
    i = 0
    while i < n:
        yield i          # pause here, hand out i, resume on next request
        i += 1

gen = count_up(3)
print(gen)                            # -> <generator object count_up at 0x...>
print(list(gen))                      # -> [0, 1, 2]
print(list(gen))                      # -> []   generators are used up after one pass!

# ---------------------------------------------------------------
# 3. A mini "DataLoader" using a generator
# ---------------------------------------------------------------
def batches(data, batch_size):
    """Yield successive chunks of `data` of size `batch_size`."""
    for start in range(0, len(data), batch_size):
        yield data[start:start + batch_size]

dataset = list(range(10))
for i, batch in enumerate(batches(dataset, batch_size=4)):
    print(f"batch {i}: {batch}")
# -> batch 0: [0, 1, 2, 3]
# -> batch 1: [4, 5, 6, 7]
# -> batch 2: [8, 9]         (the last batch can be smaller)

# ---------------------------------------------------------------
# 4. Memory: list vs generator
# ---------------------------------------------------------------
import sys
big_list = [x for x in range(100_000)]
big_gen = (x for x in range(100_000))
print(sys.getsizeof(big_list) > 100_000)   # -> True   (hundreds of KB)
print(sys.getsizeof(big_gen) < 500)        # -> True   (tiny, values made on demand)

# ---------------------------------------------------------------
# 5. Infinite generator (stop it yourself)
# ---------------------------------------------------------------
def lr_schedule(start, decay):
    lr = start
    while True:
        yield lr
        lr *= decay

sched = lr_schedule(0.1, 0.5)
print([round(next(sched), 4) for _ in range(4)])   # -> [0.1, 0.05, 0.025, 0.0125]

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Write a generator fib() that yields Fibonacci numbers forever.
#    Print the first 10 using next() in a loop.
# 2. Change batches() so it shuffles the data first (random.shuffle on a copy).
