import numpy as np

class LambdaIteration(object):
    def __init__(self):
        self.lambda_value = None
        self.history = {}
        self.systemLoad = None

    def solve(self, system, load, MAX_ITER=15):
        self.history = {"lambda": [], "err": []}
        self.systemLoad = load
        min_power = np.array([u.min_power for u in system])
        max_power = np.array([u.max_power for u in system])
        lambda_min = min(unit.marginal_cost(unit.min_power)
                        for unit in system.units)
        lambda_max = max(unit.marginal_cost(unit.max_power)
                        for unit in system.units)
        Delta = (lambda_max - lambda_min) / 2
        self.lambda_value = lambda_min + Delta
        eps = 1
        for i in range(MAX_ITER):
            powers = np.array([u.inv_marginal_cost(self.lambda_value)
                               for u in system])
            # clip
            powers = np.minimum(max_power, np.maximum(min_power, powers))
            eps = sum(powers) - load

            self.history['lambda'].append(self.lambda_value)
            self.history['err'].append(eps)

            Delta /= 2
            if eps > 0:
                self.lambda_value -= Delta
            if eps < 0:
                self.lambda_value += Delta
            if (abs(eps) < 1e-3):
                break
            if np.linalg.norm(powers - min_power) < 5e-2 or np.linalg.norm(powers - max_power) < 5e-2:
                break
            
        return powers
