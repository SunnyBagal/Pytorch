"""
Lesson 17 — Decorators and Context Managers
===========================================

Two pieces of syntax you WILL see in PyTorch code:

    @torch.no_grad()              <- a DECORATOR
    def evaluate(model): ...

    with torch.no_grad():         <- a CONTEXT MANAGER
        preds = model(x)

You don't need to write many yourself, but you should understand them.

Run:  python basics/17_decorators_and_context_managers.py
"""

import time
from contextlib import contextmanager
from functools import wraps

# ---------------------------------------------------------------
# 1. Decorators — wrap a function to add behavior
# ---------------------------------------------------------------
# A decorator is a function that takes a function and returns a new one.
def shout(fn):
    @wraps(fn)                         # keeps the original name/docstring
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        return result.upper()
    return wrapper

@shout                                 # same as: greet = shout(greet)
def greet(name):
    return f"hello {name}"

print(greet("sunny"))                  # -> HELLO SUNNY

# A practical one: timing any function
def timer(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        print(f"{fn.__name__} took {time.perf_counter() - start:.3f}s")
        return out
    return wrapper

@timer
def slow_add(a, b):
    time.sleep(0.1)
    return a + b

print(slow_add(2, 3))
# -> slow_add took 0.10xs
# -> 5

# ---------------------------------------------------------------
# 2. Context managers — setup / teardown around a block (`with`)
# ---------------------------------------------------------------
# You already used one: `with open(...) as f:` closes the file for you.
# Make your own with @contextmanager: code before `yield` = setup,
# code after `yield` = teardown (runs even if an error happens).
@contextmanager
def training_mode_off(model_state):
    old = model_state["training"]
    model_state["training"] = False    # setup
    try:
        yield model_state
    finally:
        model_state["training"] = old  # teardown: restore

state = {"training": True}
with training_mode_off(state):
    print("inside:", state)            # -> inside: {'training': False}
print("after: ", state)                # -> after:  {'training': True}

# That's conceptually what `with torch.no_grad():` does: temporarily turn
# gradient tracking OFF, then restore it afterwards.

# ---------------------------------------------------------------
# 3. Class-based context manager (__enter__ / __exit__)
# ---------------------------------------------------------------
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start

with Timer() as t:
    sum(range(100_000))
print(t.elapsed > 0)                   # -> True

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Write a decorator @log_calls that prints the function name and its
#    arguments every time it's called.
# 2. Write a context manager that prints "start" and "end" around a block.
