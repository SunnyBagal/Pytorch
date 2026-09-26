"""
Lesson 13 — Errors and Exceptions
=================================

When something goes wrong, Python RAISES an EXCEPTION. If nobody handles
it, the program stops and prints a "traceback".

Learning to READ errors is a superpower. Read tracebacks from the BOTTOM:
the last line tells you the error type and message.

Common ones you'll meet in PyTorch:
  RuntimeError   -> shape mismatch, e.g. "mat1 and mat2 shapes cannot be multiplied"
  TypeError      -> wrong kind of argument
  IndexError     -> index out of range
  KeyError       -> missing dict key (e.g. when loading a state_dict)

Run:  python basics/13_exceptions.py
"""

# ---------------------------------------------------------------
# 1. try / except
# ---------------------------------------------------------------
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print("Caught:", e)             # -> Caught: division by zero

# ---------------------------------------------------------------
# 2. Catching different error types
# ---------------------------------------------------------------
def safe_get(items, idx):
    try:
        return items[idx]
    except IndexError:
        return "index out of range"
    except TypeError:
        return "index must be an int"

print(safe_get([1, 2, 3], 1))       # -> 2
print(safe_get([1, 2, 3], 10))      # -> index out of range
print(safe_get([1, 2, 3], "a"))     # -> index must be an int

# ---------------------------------------------------------------
# 3. else and finally
# ---------------------------------------------------------------
try:
    n = int("42")
except ValueError:
    print("not a number")
else:
    print("converted:", n)          # -> converted: 42   (runs only if no error)
finally:
    print("always runs")            # -> always runs     (cleanup code)

# ---------------------------------------------------------------
# 4. Raising your own errors
# ---------------------------------------------------------------
def set_learning_rate(lr):
    if lr <= 0:
        raise ValueError(f"learning rate must be positive, got {lr}")
    return lr

try:
    set_learning_rate(-0.1)
except ValueError as e:
    print(e)                        # -> learning rate must be positive, got -0.1

# ---------------------------------------------------------------
# 5. assert — quick sanity checks while developing
# ---------------------------------------------------------------
shape = (32, 3, 28, 28)
assert len(shape) == 4, "expected a 4D batch"      # passes silently

try:
    assert shape[1] == 1, f"expected 1 channel, got {shape[1]}"
except AssertionError as e:
    print("AssertionError:", e)     # -> AssertionError: expected 1 channel, got 3

# ---------------------------------------------------------------
# 6. Don't hide bugs
# ---------------------------------------------------------------
# BAD:   try: ... except: pass      (swallows every error silently)
# GOOD:  catch the SPECIFIC error you expect, and handle it.

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Write a function that asks for input() and keeps asking until the
#    user types a valid integer (use while True + try/except ValueError).
# 2. Deliberately cause a KeyError and read the traceback carefully.
