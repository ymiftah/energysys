class Unit(object):
    def __init__(self, name):
        self.name = name

class System(object):
    def __init__(self, name, units):
        self.name = name
        self.units = units

    def __iter__(self):
        for u in self.units:
            yield u