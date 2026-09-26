"""
Lesson 08 — Functions
=====================

A FUNCTION is a named, reusable block of code. You define it with `def`.
It can take inputs (PARAMETERS) and give back an output with `return`.

In PyTorch you'll write functions like train_one_epoch(), evaluate(), etc.

Run:  python basics/08_functions.py
"""

# ---------------------------------------------------------------
# 1. Basic function
# ---------------------------------------------------------------
def greet(name):
    """Docstring: describes what the function does (shows up in help())."""
    return f"Hello, {name}!"

print(greet("Sunny"))         # -> Hello, Sunny!

# A function without `return` returns None
def say_hi():
    print("hi")

result = say_hi()             # -> hi
print(result)                 # -> None

# ---------------------------------------------------------------
# 2. Default and keyword arguments
# ---------------------------------------------------------------
def train(epochs, lr=0.01, verbose=False):
    return f"epochs={epochs}, lr={lr}, verbose={verbose}"

print(train(5))                        # -> epochs=5, lr=0.01, verbose=False
print(train(5, 0.1))                   # -> epochs=5, lr=0.1, verbose=False
print(train(epochs=3, verbose=True))   # -> epochs=3, lr=0.01, verbose=True
# Keyword arguments (name=value) make calls readable — PyTorch uses them a lot:
#   nn.Linear(in_features=10, out_features=1)

# ---------------------------------------------------------------
# 3. Returning multiple values (a tuple)
# ---------------------------------------------------------------
def stats(values):
    mean = sum(values) / len(values)
    return min(values), max(values), mean

lo, hi, avg = stats([2, 4, 6, 8])
print(lo, hi, avg)            # -> 2 8 5.0

# ---------------------------------------------------------------
# 4. *args and **kwargs — accept any number of arguments
# ---------------------------------------------------------------
def add_all(*args):           # args is a TUPLE of positional arguments
    return sum(args)

print(add_all(1, 2, 3, 4))    # -> 10

def show_config(**kwargs):    # kwargs is a DICT of keyword arguments
    for k, v in kwargs.items():
        print(f"{k}={v}")

show_config(lr=0.001, batch_size=64)
# -> lr=0.001
# -> batch_size=64

# Unpacking a dict INTO a function call with **
params = {"epochs": 10, "lr": 0.5}
print(train(**params))        # -> epochs=10, lr=0.5, verbose=False

# ---------------------------------------------------------------
# 5. Scope: variables inside a function are LOCAL
# ---------------------------------------------------------------
counter = 0
def increment():
    counter = 100             # this is a NEW local variable
    return counter

print(increment(), counter)   # -> 100 0   (global counter unchanged)

# ---------------------------------------------------------------
# 6. Type hints (optional but helpful)
# ---------------------------------------------------------------
def accuracy(correct: int, total: int) -> float:
    return correct / total

print(accuracy(45, 50))       # -> 0.9
# Hints don't enforce anything at runtime; they help you and your editor.

# ---------------------------------------------------------------
# 7. Functions are objects — you can pass them around
# ---------------------------------------------------------------
def square(x):
    return x * x

def apply(fn, value):
    return fn(value)

print(apply(square, 6))       # -> 36
# PyTorch does this too: you pass an activation function or loss function
# as an argument.

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Write mse(preds, targets) that returns the mean squared error of two
#    lists: average of (p - t)**2. Test: mse([1,2,3],[1,2,5]) -> 1.333...
# 2. Give it a default argument `reduction="mean"` that can also be "sum".
