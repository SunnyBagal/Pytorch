"""
Lesson 11 — Classes and Objects (OOP)
=====================================

This is THE most important Python lesson for PyTorch, because every
neural network you build is a CLASS.

CLASS  = a blueprint (e.g. "Dog").
OBJECT = an instance built from the blueprint (e.g. my_dog = Dog("Rex")).
ATTRIBUTES = data stored on the object (self.name).
METHODS    = functions that belong to the class (def bark(self)).

Run:  python basics/11_classes_and_objects.py
"""

# ---------------------------------------------------------------
# 1. A simple class
# ---------------------------------------------------------------
class Dog:
    species = "Canis familiaris"      # CLASS attribute: shared by all dogs

    def __init__(self, name, age):
        # __init__ runs automatically when you create an object.
        # `self` is the object being created. Store data on it.
        self.name = name              # INSTANCE attribute: unique per dog
        self.age = age

    def bark(self):
        return f"{self.name} says woof!"

    def birthday(self):
        self.age += 1                 # methods can change the object's state


rex = Dog("Rex", 3)
bella = Dog("Bella", 5)

print(rex.name, rex.age)          # -> Rex 3
print(bella.bark())               # -> Bella says woof!
rex.birthday()
print(rex.age)                    # -> 4
print(rex.species)                # -> Canis familiaris

# ---------------------------------------------------------------
# 2. Special ("dunder") methods: __str__, __len__, __call__, ...
# ---------------------------------------------------------------
# Methods with double underscores let your objects work with Python's
# built-in syntax.
class Batch:
    def __init__(self, items):
        self.items = items

    def __len__(self):                # makes len(batch) work
        return len(self.items)

    def __getitem__(self, idx):       # makes batch[idx] work
        return self.items[idx]

    def __repr__(self):               # how the object is shown when printed
        return f"Batch(size={len(self)})"

b = Batch([10, 20, 30])
print(len(b))      # -> 3
print(b[1])        # -> 20
print(b)           # -> Batch(size=3)

# PyTorch's Dataset class uses exactly __len__ and __getitem__!

# ---------------------------------------------------------------
# 3. __call__ — make an object callable like a function
# ---------------------------------------------------------------
class Scaler:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor

double = Scaler(2)
print(double(21))  # -> 42

# In PyTorch you write `output = model(x)`. That works because nn.Module
# defines __call__, which runs your forward() method.

# ---------------------------------------------------------------
# 4. A mini "neuron" class — preview of what PyTorch does for you
# ---------------------------------------------------------------
class Neuron:
    """y = w * x + b"""

    def __init__(self, w, b):
        self.w = w      # weight
        self.b = b      # bias

    def forward(self, x):
        return self.w * x + self.b

    def __call__(self, x):
        return self.forward(x)

n = Neuron(w=2.0, b=1.0)
print(n(3.0))       # -> 7.0   (2*3 + 1)
print([n(x) for x in [0, 1, 2]])   # -> [1.0, 3.0, 5.0]

# ---------------------------------------------------------------
# 5. Checking types
# ---------------------------------------------------------------
print(isinstance(rex, Dog))       # -> True
print(hasattr(rex, "age"))        # -> True

# ---------------------------------------------------------------
# PRACTICE
# ---------------------------------------------------------------
# 1. Create a BankAccount class with deposit(amount), withdraw(amount)
#    and a __repr__ that shows the balance.
# 2. Add a __len__ to Neuron's cousin: a Layer that holds a list of Neurons.
