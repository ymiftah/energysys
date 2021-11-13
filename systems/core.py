from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

class Unit(object):
    def __init__(self, name):
        self.name = name

class System(object):
    def __init__(self, units):
        self.units = {u.name: u for u in units}

    def __iter__(self):
        for u in self.units.values():
            yield u

    def __getitem__(self, key):
        return self.units[key]

    def plot_input_output(self):
        for name, u in self.units.items():   
            pp = np.linspace(u.min_power, u.max_power, 30)
            plt.plot(pp, u.input_output(pp), label=name)
        plt.legend()
        plt.xlabel('Power [kW]')
        plt.ylabel('Fuel Consumption [kWh]')

class UCSystem(System):
    def __init__(self, units, reserve_req=0):
        super().__init__(units)
        self.reserve_req = reserve_req


class Line(object):
    def __init__(self, arc, power_lim=None, Z=100):
        self.arc = arc
        self.power_lim = power_lim
        self.Z = Z
    
    def __str__(self):
        return "{}-{}".format(*self.arc)
    
    def __repr__(self):
        return "(Line {}-{}: Plim={}, Z={})".format(*self.arc, self.power_lim, self.Z)

    def tuple(self):
        return self.arc

class Network(object):
    def __init__(self, lines, system, links):
        self.lines = {l.tuple(): l for l in lines}
        self.system = system
        self.buses = sorted(list(reduce(lambda a, b : a.union(b),
                            [set(l.arc) for l in lines])))
        self.links = set(links)

    def __iter__(self):
        for b in self.buses:
            yield b

    def __getitem__(self, key):
        return self.lines[key]
    
    def display(self):
        for l in self.lines:
            print(l)

    def link(self, bus, unit):
        return (bus, unit) in self.links
    
    def power_lim(self, a, b):
        line = self.lines.get((a,b), self.lines.get((b,a), None))
        if line is None:
            return None
        else:
            return line.power_lim
    
    def Z(self, a, b):
        line = self.lines.get((a,b), self.lines.get((b,a), None))
        if line is None:
            return 1
        else:
            return line.Z