"""
Lesson 02 — Strings
===================

A STRING (str) is a sequence of characters. You'll use strings for file
paths, labels, printing training progress, and processing text data.

Strings are IMMUTABLE: methods return a NEW string, the original never changes.

Run:  python basics/02_strings.py
"""

text = "Hello, PyTorch"

# ---------------------------------------------------------------
# 1. Length and indexing (positions start at 0!)
# ---------------------------------------------------------------
print(len(text))     # -> 14      number of characters
print(text[0])       # -> H       first character
print(text[-1])      # -> h       negative index counts from the end

# ---------------------------------------------------------------
# 2. Slicing: text[start:stop:step]   (stop is NOT included)
# ---------------------------------------------------------------
print(text[0:5])     # -> Hello
print(text[7:])      # -> PyTorch   (from 7 to the end)
print(text[:5])      # -> Hello     (from the start to 5)
print(text[::-1])    # -> hcroTyP ,olleH   (step -1 reverses)

# Slicing works the SAME way on lists, NumPy arrays and PyTorch tensors,
# so learn it well here.

# ---------------------------------------------------------------
# 3. Useful string methods
# ---------------------------------------------------------------
print(text.upper())                    # -> HELLO, PYTORCH
print(text.lower())                    # -> hello, pytorch
print(text.replace("PyTorch", "AI"))   # -> Hello, AI
print(text.split(", "))                # -> ['Hello', 'PyTorch']   str -> list
print("-".join(["a", "b", "c"]))       # -> a-b-c                  list -> str
print("  padded  ".strip())            # -> padded   (removes spaces at both ends)
print(text.startswith("Hello"))        # -> True
print(text.find("Py"))                 # -> 7   (index where it starts, -1 if missing)
print("torch" in text.lower())         # -> True   `in` checks membership

# ---------------------------------------------------------------
# 4. Combining strings
# ---------------------------------------------------------------
first, last = "Sunny", "Bagal"
print(first + " " + last)     # -> Sunny Bagal    (+ concatenates)
print("ab" * 3)               # -> ababab         (* repeats)

# f-string formatting you'll use when printing training logs:
epoch, loss, acc = 3, 0.245678, 0.9312
print(f"Epoch {epoch:02d} | loss={loss:.4f} | acc={acc:.1%}")
# -> Epoch 03 | loss=0.2457 | acc=93.1%

# ---------------------------------------------------------------
# 5. Multi-line strings
# ---------------------------------------------------------------
poem = """Line one
Line two"""
print(poem)
# -> Line one
# -> Line two

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Given s = "deep learning is fun", print it in Title Case (hint: .title()).
# 2. Count how many words it has (hint: split + len).
# 3. Check if a word is a palindrome using slicing.
