
class Base():

    def __init__(self):
        self.attr = None
        print(self.attr)

    def print(self):
        print(self.attr)

class Child(Base):
    def __init__(self):
        super().__init__()
        self.attr = "attr from child"
        print(self.attr)

    def print1(self):
        print(self.attr)


base = Base()
child = Child()
base.print()
child.print()
child.print1()
