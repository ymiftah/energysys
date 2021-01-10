import numpy as np

import pyomo.environ as pyo
from scipy.optimize import Bounds, LinearConstraint, minimize


class QPScipy(object):
    def __init__(self):
        res = None

    def solve(self, system, load, tee=0):
        # Variables bounds
        bounds = Bounds([u.min_power for u in system], [u.max_power for u in system])
        # Starting point
        x0 = np.array([u.min_power for u in system])
        # Objective function
        def fobj(x):
            return sum(u.input_output(x[i]) * u.fuel_cost for i, u in enumerate(system))
        # Jacobian
        def fobj_jac(x):
            return np.array([u.marginal_heatrate(x[i]) * u.fuel_cost for i, u in enumerate(system)])
        # Constraint
        balance = LinearConstraint(np.ones(len(x0)), [load], [np.inf])
        self.res = minimize(fobj, x0, method='trust-constr', jac=fobj_jac,
                constraints=[balance], options={'verbose': 1}, bounds=bounds)



class QPModel(object):
    def __init__(self):
        self.m = pyo.ConcreteModel()

    def solve(self, system, load, tee=0):
        self.m.units = pyo.Set(initialize=[u.name for u in system])
        lo = {u.name: u.min_power for u in system}
        up = {u.name: u.max_power for u in system}
        def bounds(model, i):
            return (lo[i], up[i])
        self.m.varPower = pyo.Var(self.m.units, bounds=bounds)
        self.m.cost = pyo.Objective(
            expr = sum(u.input_output(self.m.varPower[u.name]) * u.fuel_cost for u in system),
            sense = pyo.minimize,
        )
        self.m.balance = pyo.Constraint(rule= lambda m: load <= sum(m.varPower[u] for u in m.units))
        sol = pyo.SolverFactory("ipopt", executable="./solvers/ipopt/ipopt.exe")
        res = sol.solve(self.m, tee=tee)
        for v in self.m.component_data_objects(pyo.Var, active=True):
            print(v, pyo.value(v))  # doctest: +SKIP
        print(pyo.value(self.m.cost))



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
