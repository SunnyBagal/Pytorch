"""
Lesson 07 — Loops (for / while)
===============================

Loops repeat code. Training a neural network IS a loop:
    for each epoch:
        for each batch:
            predict -> compute loss -> update weights

Run:  python basics/07_loops.py
"""

# ---------------------------------------------------------------
# 1. for loop over a list
# ---------------------------------------------------------------
for fruit in ["apple", "banana", "cherry"]:
    print(fruit)
# -> apple
# -> banana
# -> cherry

# ---------------------------------------------------------------
# 2. range(start, stop, step) — generates numbers (stop NOT included)
# ---------------------------------------------------------------
print(list(range(5)))          # -> [0, 1, 2, 3, 4]
print(list(range(2, 10, 3)))   # -> [2, 5, 8]

total = 0
for i in range(1, 6):
    total += i                 # same as total = total + i
print(total)                   # -> 15

# ---------------------------------------------------------------
# 3. enumerate() — get the index AND the item
# ---------------------------------------------------------------
losses = [0.9, 0.6, 0.4]
for epoch, loss in enumerate(losses, start=1):
    print(f"epoch {epoch}: loss {loss}")
# -> epoch 1: loss 0.9
# -> epoch 2: loss 0.6
# -> epoch 3: loss 0.4

# ---------------------------------------------------------------
# 4. zip() — loop over several lists side by side
# ---------------------------------------------------------------
images = ["img1.png", "img2.png", "img3.png"]
labels = ["cat", "dog", "cat"]
for img, lbl in zip(images, labels):
    print(img, "->", lbl)
# -> img1.png -> cat
# -> img2.png -> dog
# -> img3.png -> cat

# ---------------------------------------------------------------
# 5. while loop — repeat while a condition is True
# ---------------------------------------------------------------
value = 100.0
steps = 0
while value > 1:
    value /= 2
    steps += 1
print(steps, value)            # -> 7 0.78125

# ---------------------------------------------------------------
# 6. break and continue
# ---------------------------------------------------------------
for n in range(10):
    if n % 2 == 0:
        continue               # skip the rest of this iteration
    if n > 7:
        break                  # exit the loop completely
    print(n, end=" ")          # end=" " prints on one line
print()                        # newline
# -> 1 3 5 7

# Early stopping — a real ML pattern using break:
val_losses = [0.8, 0.6, 0.55, 0.56, 0.57, 0.58]
best, patience, bad_epochs = float("inf"), 2, 0
for epoch, vl in enumerate(val_losses):
    if vl < best:
        best, bad_epochs = vl, 0
    else:
        bad_epochs += 1
    if bad_epochs >= patience:
        print(f"early stop at epoch {epoch}, best={best}")
        break
# -> early stop at epoch 4, best=0.55

# ---------------------------------------------------------------
# 7. Nested loops
# ---------------------------------------------------------------
for epoch in range(2):
    for batch in range(3):
        print(f"e{epoch}b{batch}", end=" ")
print()
# -> e0b0 e0b1 e0b2 e1b0 e1b1 e1b2

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Print the multiplication table of 7 (7 x 1 ... 7 x 10).
# 2. Loop over two lists `preds` and `targets` with zip and count
#    how many match -> that's accuracy!
