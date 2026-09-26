"""
Lesson 12 — Inheritance and super()
===================================

INHERITANCE lets a new class (child) reuse and extend an existing class
(parent). The child gets all the parent's methods for free.

In PyTorch you ALWAYS do this:

    class MyNet(nn.Module):          # MyNet inherits from nn.Module
        def __init__(self):
            super().__init__()       # run the parent's setup first!
            ...

This lesson explains exactly what that means.

Run:  python basics/12_inheritance.py
"""

# ---------------------------------------------------------------
# 1. Parent and child
# ---------------------------------------------------------------
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "..."

    def describe(self):
        return f"{self.name} says {self.speak()}"


class Cat(Animal):                  # Cat IS-A Animal
    def speak(self):                # OVERRIDE the parent's method
        return "meow"


class Cow(Animal):
    def speak(self):
        return "moo"


for a in [Cat("Tom"), Cow("Daisy"), Animal("???")]:
    print(a.describe())             # describe() comes from Animal
# -> Tom says meow
# -> Daisy says moo
# -> ??? says ...

# ---------------------------------------------------------------
# 2. super() — call the parent's version of a method
# ---------------------------------------------------------------
class Model:
    def __init__(self):
        self.training = True        # the parent sets up some state
        self.layers = []
        print("Model.__init__ ran")

    def eval(self):
        self.training = False


class TinyNet(Model):
    def __init__(self, hidden):
        super().__init__()          # run Model.__init__ -> self.training, self.layers
        self.hidden = hidden        # then add our own stuff
        self.layers.append(f"Linear(2->{hidden})")


net = TinyNet(hidden=8)             # -> Model.__init__ ran
print(net.layers, net.training)     # -> ['Linear(2->8)'] True
net.eval()                          # inherited method
print(net.training)                 # -> False

# What if we forget super().__init__()?
class BrokenNet(Model):
    def __init__(self):
        self.hidden = 4             # parent setup never ran!

broken = BrokenNet()
print(hasattr(broken, "layers"))    # -> False  -> would crash later.
# In PyTorch, forgetting super().__init__() gives the error:
#   "cannot assign module before Module.__init__() call"

# ---------------------------------------------------------------
# 3. isinstance works with parents too
# ---------------------------------------------------------------
print(isinstance(net, TinyNet), isinstance(net, Model))   # -> True True

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Make a Shape class with area() returning 0, then Rectangle(w, h) and
#    Circle(r) that override area(). Put them in a list and print areas.
# 2. Make a child class that calls super().describe() and adds extra text.
