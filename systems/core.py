class Unit(object):
    def __init__(self, name):
        self.name = name

class System(object):
    def __init__(self, name, units):
        self.name = name
        self.units = {u.name: u for u in units}

    def __iter__(self):
        for u in self.units.values():
            yield u

    def __getitem__(self, key):
        return self.units[key]