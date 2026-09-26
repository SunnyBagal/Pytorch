"""
Lesson 03 — Lists
=================

A LIST is an ordered, changeable (mutable) collection. It can hold any types.
Lists are everywhere in ML code: a list of losses per epoch, a list of
file names, a list of layers, a batch of samples...

Run:  python basics/03_lists.py
"""

# ---------------------------------------------------------------
# 1. Creating and accessing
# ---------------------------------------------------------------
nums = [10, 20, 30, 40, 50]
mixed = [1, "two", 3.0, True]

print(nums[0], nums[-1])   # -> 10 50
print(nums[1:4])           # -> [20, 30, 40]   (same slicing as strings)
print(len(nums))           # -> 5

# ---------------------------------------------------------------
# 2. Changing a list (lists are MUTABLE)
# ---------------------------------------------------------------
nums[0] = 99               # replace an item
print(nums)                # -> [99, 20, 30, 40, 50]

nums.append(60)            # add ONE item at the end
print(nums)                # -> [99, 20, 30, 40, 50, 60]

nums.extend([70, 80])      # add MANY items at the end
print(nums)                # -> [99, 20, 30, 40, 50, 60, 70, 80]

nums.insert(1, 15)         # insert 15 at index 1
print(nums)                # -> [99, 15, 20, 30, 40, 50, 60, 70, 80]

last = nums.pop()          # remove & return the LAST item
print(last, nums)          # -> 80 [99, 15, 20, 30, 40, 50, 60, 70]

nums.remove(99)            # remove the first item equal to 99
print(nums)                # -> [15, 20, 30, 40, 50, 60, 70]

# ---------------------------------------------------------------
# 3. Sorting and useful built-ins
# ---------------------------------------------------------------
scores = [3, 1, 4, 1, 5, 9, 2]
print(sorted(scores))              # -> [1, 1, 2, 3, 4, 5, 9]  (returns a NEW list)
print(sorted(scores, reverse=True))# -> [9, 5, 4, 3, 2, 1, 1]
print(sum(scores), min(scores), max(scores))  # -> 25 1 9
print(scores.count(1))             # -> 2   how many times 1 appears
print(scores.index(5))             # -> 4   position of the first 5
scores.sort()                      # sorts IN PLACE (changes scores, returns None)
print(scores)                      # -> [1, 1, 2, 3, 4, 5, 9]

# ---------------------------------------------------------------
# 4. The aliasing trap (VERY important)
# ---------------------------------------------------------------
a = [1, 2, 3]
b = a            # b is NOT a copy — both names point to the SAME list
b.append(4)
print(a)         # -> [1, 2, 3, 4]   a changed too!

c = a.copy()     # make a real copy (also: list(a) or a[:])
c.append(5)
print(a, c)      # -> [1, 2, 3, 4] [1, 2, 3, 4, 5]

# PyTorch tensors behave the same way; there you use tensor.clone() to copy.

# ---------------------------------------------------------------
# 5. Nested lists (a "matrix")
# ---------------------------------------------------------------
matrix = [[1, 2, 3],
          [4, 5, 6]]
print(matrix[1])      # -> [4, 5, 6]   second row
print(matrix[1][2])   # -> 6           row 1, column 2
# Doing math on nested lists is slow and clumsy -> that's why NumPy
# arrays and PyTorch tensors exist.

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Make a list of 5 losses, e.g. [0.9, 0.7, 0.5, 0.4, 0.35].
#    Print the average loss (sum / len) and the best (lowest) one.
# 2. Append a new loss and print only the last 3 losses using slicing.
