"""
Lesson 06 — Conditionals (if / elif / else)
===========================================

Conditionals let your program make decisions. Python uses INDENTATION
(4 spaces) to mark which lines belong to which block — no curly braces.

Run:  python basics/06_conditionals.py
"""

# ---------------------------------------------------------------
# 1. Comparison operators -> produce True/False
# ---------------------------------------------------------------
x = 7
print(x > 5, x < 5, x == 7, x != 7, x >= 7, x <= 6)
# -> True False True False True False

# ---------------------------------------------------------------
# 2. if / elif / else
# ---------------------------------------------------------------
accuracy = 0.87

if accuracy >= 0.9:
    grade = "excellent"
elif accuracy >= 0.8:          # checked only if the first test was False
    grade = "good"
else:                          # runs if nothing above matched
    grade = "needs work"
print(grade)                   # -> good

# ---------------------------------------------------------------
# 3. Logical operators: and, or, not
# ---------------------------------------------------------------
loss, epoch = 0.05, 12
if loss < 0.1 and epoch > 10:
    print("stop training")     # -> stop training

has_gpu = False
print(not has_gpu)             # -> True
print(has_gpu or epoch > 5)    # -> True

# Chained comparisons (Pythonic!)
lr = 0.001
print(0 < lr < 1)              # -> True   same as (0 < lr) and (lr < 1)

# ---------------------------------------------------------------
# 4. Truthiness: empty things are False
# ---------------------------------------------------------------
batch = []
if not batch:
    print("batch is empty")    # -> batch is empty

name = "resnet"
if name:
    print(f"model = {name}")   # -> model = resnet

# ---------------------------------------------------------------
# 5. One-line "ternary" expression
# ---------------------------------------------------------------
device = "cuda" if has_gpu else "cpu"
print(device)                  # -> cpu
# You will write EXACTLY this pattern in PyTorch:
#   device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------
# 6. `is` vs `==`
# ---------------------------------------------------------------
# ==  compares VALUES,  `is` checks if it's the SAME object.
# Use `is` only for None:
result = None
if result is None:
    print("no result yet")     # -> no result yet

# ---------------------------------------------------------------
# 7. match (Python 3.10+) — a cleaner multi-way branch
# ---------------------------------------------------------------
optimizer = "adam"
match optimizer:
    case "sgd":
        print("plain SGD")
    case "adam" | "adamw":
        print("Adam family")   # -> Adam family
    case _:
        print("unknown")

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Given a number n, print "even" or "odd" (hint: n % 2).
# 2. Given a probability p, print "positive" if p >= 0.5 else "negative"
#    using a one-line ternary expression.
