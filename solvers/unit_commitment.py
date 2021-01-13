from solvers.economic_dispatch import QPModel
import numpy as np
import pandas as pd

from pyomo.environ import *


class QPModel(object):
    def __init__(self):
        self.m = None
        self.built = False

    def build_model(self, system, load):
        r = system.reserve_req

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

        def eq_min_uptime(m, t, u):
            if system[u].min_uptime is None or t > system[u].min_uptime:
                return Constraint.Skip
            else:
                return (m.binShutDown[t, u] <=
                    1 - sum(m.binStartUp[tt, u] for tt in m.t
                        if t - system[u].min_uptime <= tt <= t)    
                )
        m.eq_min_uptime = Constraint(m.t, m.units, rule=eq_min_uptime)

        def eq_min_rest(m, t, u):
            if system[u].min_rest is None or t > system[u].min_rest:
                return Constraint.Skip
            else:
                return (m.binStartUp[t, u] <=
                    1 - sum(m.binShutDown[tt, u] for tt in m.t
                        if t - system[u].min_rest <= tt <= t)    
                )
        m.eq_min_rest = Constraint(m.t, m.units, rule=eq_min_rest)
            
        # Ramp Equations
        def eq_ramp_up(m, t, u):
            if  system[u].ramp_up is None or t == m.t.first():
                return Constraint.Skip
            else:
                return m.varPower[t, u] - m.varPower[t-1, u] <= system[u].ramp_up * m.binIsOn[t-1,u] + system[u].min_power * m.binStartUp[t,u]
        
        def eq_ramp_down(m, t, u):
            if  system[u].ramp_down is None or t == m.t.first():
                return Constraint.Skip
            else:
                return m.varPower[t-1, u] - m.varPower[t, u] <= system[u].ramp_down * m.binIsOn[t,u] + system[u].min_power * m.binShutDown[t,u]
        m.eq_ramp_up = Constraint(m.t, m.units, rule=eq_ramp_up)
        m.eq_ramp_down = Constraint(m.t, m.units, rule=eq_ramp_down)

        # Min Max Power
        def eq_min_power(m, t, u):
            return m.varPower[t, u] >= system[u].min_power * m.binIsOn[t, u]
        def eq_max_power(m, t, u):
            return m.varPower[t, u] <= system[u].max_power * m.binIsOn[t, u]
        m.eq_min_power = Constraint(m.t, m.units, rule=eq_min_power)
        m.eq_max_power = Constraint(m.t, m.units, rule=eq_max_power)

        # Reserves
        if 1 > r > 0:
            m.varReserve = Var(m.t, m.units, domain=NonNegativeReals)
            # Power reserve
            m.eq_balance_reserve = Constraint(m.t, 
                rule=lambda m, t: sum(m.varReserve[t,u] for u in m.units) >= load[t]* (1+r))
            # Reserve > Power
            m.eq_reserve_power = Constraint(m.t, m.units, 
                rule=lambda m, t, u: m.varReserve[t,u] >= m.varPower[t,u])
            # Reserve < Max Power
            m.eq_reserve_max_power = Constraint(m.t, m.units, 
                rule=lambda m, t, u: m.varReserve[t,u] <= system[u].max_power * m.binIsOn[t, u])
            # Reserve ramp up
            m.eq_reserve_ramp_up = Constraint(m.t, m.units, 
                rule=lambda m, t, u: (
                Constraint.Skip if (system[u].ramp_up is None or t == m.t.first())
                else m.varReserve[t, u] <= m.varPower[t-1, u] + system[u].ramp_up * m.binIsOn[t-1,u] + system[u].min_power * m.binStartUp[t,u]
                )
            )
            # Reserve ramp down
            m.eq_reserve_ramp_up = Constraint(m.t, m.units, 
                rule=lambda m, t, u: (
                Constraint.Skip if (system[u].ramp_up is None or t == m.t.last())
                else m.varReserve[t, u] <= system[u].min_power * m.binShutDown[t+1,u] + system[u].min_power * (m.binIsOn[t,u] - m.binShutDown[t+1,u])  
                )
            )

        # Update 
        self.m = m
        self.built = True
        
        
    def solve(self, system, load, tee=0, force_build=False):
        if not self.built or force_build: 
            self.build_model(system, load)    
        sol = SolverFactory("cplex")
        sol.options["mipgap"] = 0.01
        sol.options["time"] = 200
        res = sol.solve(self.m, tee=tee)
        return value(self.m.cost)


class LPModel(QPModel):
    def __init__(self):
        super().__init__()

    def build_model(self, system, load):
        super().build_model(system, load)
        m = self.m
        m.del_component(m.cost)

        m.varFuelCons = Var(m.t, m.units, domain=NonNegativeReals)
        m.support_lines = ConstraintList()
        num_lines = 2
        for t in m.t:
            for u in m.units:
                p = np.linspace(system[u].min_power, system[u].max_power, num_lines+1)
                fp = system[u].input_output(p)
                for i in range(num_lines):
                    m.support_lines.add(
                        m.varFuelCons[t, u] / system[u].fuel_cost
                        >=
                        fp[i] * m.binIsOn[t,u]
                        + (fp[i+1] - fp[i])/(p[i+1]-p[i]) * (m.varPower[t, u] - p[i])
                        )
        m.cost = Objective(
            expr = sum(m.varFuelCons[t, u.name] for u in system for t in m.t)
                   + sum(m.binStartUp[t, u.name] * u.start_up_cost for u in system for t in m.t)
                   , sense = minimize,
        )

        self.m = m
        self.built = True
        
    def solve(self, system, load, tee=0, force_build=False):
        if not self.built or force_build: 
            self.build_model(system, load)    
        sol = SolverFactory("cbc", executable="./solvers/cbc/cbc.exe")
        sol.options["ratio"] = 0.01
        sol.options["sec"] = 200
        res = sol.solve(self.m, tee=tee)
        return value(self.m.cost)
    
    def get_power(self):
        d = self.m.varPower.extract_values()
        mux = pd.MultiIndex.from_tuples(d.keys())
        df = pd.DataFrame(list(d.values()), index=mux).unstack(fill_value=0)
        df.columns = df.columns.droplevel()
        return df