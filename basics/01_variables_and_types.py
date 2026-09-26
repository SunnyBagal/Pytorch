"""
Lesson 01 — Variables and Data Types
====================================

A VARIABLE is a name that points to a value stored in memory.
You create one with `=` (assignment). No need to declare a type —
Python figures it out automatically (this is called "dynamic typing").

Run:  python basics/01_variables_and_types.py
"""

# ---------------------------------------------------------------
# 1. The core built-in types
# ---------------------------------------------------------------
age = 21                # int    -> whole numbers
height = 5.9            # float  -> decimal numbers
name = "Sunny"          # str    -> text
is_learning = True      # bool   -> True / False
nothing = None          # NoneType -> "no value yet"

# type(x) tells you what kind of value x holds.
print(type(age))          # -> <class 'int'>
print(type(height))       # -> <class 'float'>
print(type(name))         # -> <class 'str'>
print(type(is_learning))  # -> <class 'bool'>
print(type(nothing))      # -> <class 'NoneType'>

# ---------------------------------------------------------------
# 2. Arithmetic operators
# ---------------------------------------------------------------
a, b = 17, 5              # assign two variables in one line
print(a + b)    # -> 22   addition
print(a - b)    # -> 12   subtraction
print(a * b)    # -> 85   multiplication
print(a / b)    # -> 3.4  "true" division ALWAYS gives a float
print(a // b)   # -> 3    floor division (drops the decimal part)
print(a % b)    # -> 2    modulo (the remainder)
print(a ** 2)   # -> 289  power (17 squared)

# ---------------------------------------------------------------
# 3. Type conversion (casting)
# ---------------------------------------------------------------
print(int("42") + 1)      # -> 43     string -> int
print(float("3.5") * 2)   # -> 7.0    string -> float
print(str(100) + "%")     # -> 100%   int -> string
print(int(9.99))          # -> 9      float -> int TRUNCATES (doesn't round)
print(round(9.99))        # -> 10     use round() to round
print(bool(0), bool(7))   # -> False True   (0, "", [], None are "falsy")

# ---------------------------------------------------------------
# 4. f-strings: the best way to put variables inside text
# ---------------------------------------------------------------
print(f"{name} is {age} years old")        # -> Sunny is 21 years old
print(f"Height: {height:.2f} ft")          # -> Height: 5.90 ft   (2 decimals)
print(f"{a} / {b} = {a / b}")              # -> 17 / 5 = 3.4

# ---------------------------------------------------------------
# 5. Variables are labels, not boxes
# ---------------------------------------------------------------
x = 10
y = x         # y now points to the same value as x
x = 20        # x points to a NEW value; y is unaffected
print(x, y)   # -> 20 10

# Why this matters for PyTorch: tensors (and lists) are "mutable",
# so two names can point to the SAME object and changes show up in both.
# You'll see this again in lesson 03 (lists).

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Store the price of an item (float) and quantity (int). Print the total
#    using an f-string with 2 decimal places.
# 2. What does 7 // 2 give? What about -7 // 2? Predict, then run it.
