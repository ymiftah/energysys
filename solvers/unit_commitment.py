from solvers.economic_dispatch import QPModel
import numpy as np

from pyomo.environ import *


class QPModel(object):
    def __init__(self):
        self.m = None
        self.built = False

    def build_model(self, system, load):
        m = ConcreteModel()
        m.units = Set(initialize=[u.name for u in system])
        m.t = Set(initialize=list(range(len(load))), ordered=True)
        
        m.varPower = Var(m.t, m.units, domain=NonNegativeReals)
        m.binStartUp = Var(m.t, m.units, domain=Binary, initialize=0)
        m.binShutDown = Var(m.t, m.units, domain=Binary, initialize=0)
        m.binIsOn = Var(m.t, m.units, domain=Binary, initialize=0)
        m.cost = Objective(
            expr = sum(u.input_output(m.varPower[t, u.name]) * u.fuel_cost for u in system for t in m.t)
                   + sum(m.binStartUp[t, u.name] * u.start_up_cost for u in system for t in m.t)
                   + sum((m.binIsOn[t, u.name] - 1) * u.curve[0] for u in system for t in m.t)
                   , sense = minimize,
        )

        # Power balance
        m.eq_balance = Constraint(m.t, 
            rule=lambda m, t: sum(m.varPower[t, u] for u in m.units) >= load[t])
    
        # Startup Equations
        def eq_startup(m, t, u):
            if t == m.t.first():
                return Constraint.Skip
            else:
                return m.binIsOn[t, u] - m.binIsOn[t-1, u] == m.binStartUp[t, u] - m.binShutDown[t, u]
        m.eq_startup = Constraint(m.t, m.units, rule=eq_startup)

        # Min Max Power
        def eq_min_power(m, t, u):
            return m.varPower[t, u] >= system[u].min_power * m.binIsOn[t, u]
        def eq_max_power(m, t, u):
            return m.varPower[t, u] <= system[u].max_power * m.binIsOn[t, u]
        m.eq_min_power = Constraint(m.t, m.units, rule=eq_min_power)
        m.eq_max_power = Constraint(m.t, m.units, rule=eq_max_power)

        # Update 
        self.m = m
        self.built = True
        
        
    def solve(self, system, load, tee=0, force_build=False):
        if not self.built or force_build: 
            self.build_model(system, load)    
        sol = SolverFactory("bonmin", executable="./solvers/bonmin/bonmin.exe")
        sol.options["bonmin.allowable_gap"] = .05
        sol.options["bonmin.time_limit"] = 200
        res = sol.solve(self.m, tee=tee)
        for v in self.m.component_data_objects(Var, active=True):
            print(v, value(v))  # doctest: +SKIP
        print(value(self.m.cost))



class LPModel(QPModel):
    def __init__(self):
        super().__init__()

    def build_model(self, system, load):
        super().build_model(system, load)
        m = self.m
        m.del_component(m.cost)

        m.varFuelCons = Var(m.t, m.units, domain=NonNegativeReals)
        m.support_lines = ConstraintList()
        for t in m.t:
            for u in m.units:
                for i in range(3):
                    p = np.linspace(system[u].min_power, system[u].max_power, 4)
                    fp = system[u].input_output(p)
                    m.support_lines.add(
                        m.varFuelCons[t, u] / system[u].fuel_cost
                        >=
                        fp[i] * m.binIsOn[t,u]
                        + (fp[i+1] - fp[i])/(p[i+1]-p[i]) * (m.varPower[t, u] - p[i])
                        )
        m.cost = Objective(
            expr = sum(m.varFuelCons[t, u.name] for u in system for t in m.t)
                   + sum((m.binIsOn[t, u.name] - 1) * u.curve[0] for u in system for t in m.t)
                   , sense = minimize,
        )

        self.m = m
        self.built = True
        
    def solve(self, system, load, tee=0, force_build=False):
        if not self.built or force_build: 
            self.build_model(system, load)    
        sol = SolverFactory("cplex")
        sol.options["mipgap"] = 0.01
        sol.options["time"] = 200
        res = sol.solve(self.m, tee=tee)
        for v in self.m.component_data_objects(Var, active=True):
            print(v, value(v))  # doctest: +SKIP
        print(value(self.m.cost))