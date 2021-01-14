from .core import Unit, System
import numpy as np

class ThermalUnit(Unit):
    def __init__(self, name, input_output, fuel_cost,
                    min_power=-np.inf, max_power=np.inf,
                    start_up_cost=0, ramp_up=None, ramp_down=None,
                    min_uptime = None, min_rest=None):
        super().__init__(name)
        self.curve = input_output
        self.min_power = min_power
        self.max_power = max_power
        self.fuel_cost = fuel_cost
        self.start_up_cost = start_up_cost
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.min_uptime = min_uptime
        self.min_rest = min_rest
    
    def input_output(self, P):
        return (self.curve[0] + self.curve[1] * P + self.curve[2] * P * P)
    
    def net_heatrate(self, P):
        return self.input_output(P) / (P + 1e-7) * 1000
    
    def marginal_heatrate(self, P):
        return (self.curve[1] + self.curve[2] * P * 2)

    def marginal_cost(self, P):
        return self.marginal_heatrate(P) * self.fuel_cost

    def inv_marginal_cost(self, x):
        return (x / self.fuel_cost - self.curve[1]) / 2 / (self.curve[2] + 1e-7)



class UCSystem(System):
    def __init__(self, units, reserve_req=0):
        super().__init__(units)
        self.reserve_req = reserve_req
