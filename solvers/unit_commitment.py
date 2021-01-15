from solvers.economic_dispatch import QPModel
import numpy as np
import pandas as pd

from pyomo.environ import *

class BaseModel(object):
    def __init__(self):
        self.m = None

    def _build_units_equations(self, system):
        m = self.m
        m.units = Set(initialize=[u.name for u in system])
        
        m.varPower    = Var(m.t, m.units, domain=NonNegativeReals)
        m.binStartUp  = Var(m.t, m.units, domain=Binary, initialize=0)
        m.binShutDown = Var(m.t, m.units, domain=Binary, initialize=0)
        m.binIsOn     = Var(m.t, m.units, domain=Binary, initialize=0)
            
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
    def _build_reserve_equations(self, system):
        m = self.m
        r = system.reserve_req
        if r == 0:
            return
        m.varReserve = Var(m.t, m.units, domain=NonNegativeReals)
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
        self.m = m

    # Power Balance
    def _build_balance_equations(self, system, load):
        m = self.m
        # Power reserve
        m.eq_balance_power = Constraint(m.t, 
            rule=lambda m, t: sum(m.varPower[t,u] for u in m.units) >= load[t]
            )
        r = system.reserve_req
        if 1 > r > 0:
            # Power reserve
            m.eq_balance_reserve = Constraint(m.t, 
                rule=lambda m, t: sum(m.varReserve[t,u] for u in m.units) >= load[t] * (1+r)
                )
        self.m = m
        
    def _build_model(self, system, load):
        self.m   = ConcreteModel()
        self.m.t = Set(initialize=list(range(len(load))), ordered=True)
        
        
        self._build_units_equations(system)
        self._build_reserve_equations(system)
        self._build_balance_equations(system, load)


class LPModel(BaseModel):
    def __init__(self):
        super().__init__()
    
    def _build_linear_objective(self, system, num_lines=2):
        m = self.m
        m.varFuelCons = Var(m.t, m.units, domain=NonNegativeReals)
        m.support_lines = ConstraintList()
        num_lines = num_lines
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
            expr = sum(m.varFuelCons[t, u.name] * u.fuel_cost for u in system for t in m.t)
                   + sum(m.binStartUp[t, u.name] * u.start_up_cost for u in system for t in m.t)
                   , sense = minimize,
        )
        self.m = m

    def _build_model(self, system, load):
        super()._build_model(system, load)
        self._build_linear_objective(system)
        
    def solve(self, system, load, tee=0, force_build=False):
        self._build_model(system, load)    
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

class DCModel(LPModel):
    def __init__(self):
        super().__init__()

    def _build_balance_equations(self, network, load):
        m = self.m
        m.buses = Set(initialize=network.buses)
        m.arcs = Set(initialize=network.lines.keys())

        def busOut_init(m, bus):
            for i, j in m.arcs:
                if i == bus:
                    yield j
        m.busOut = Set(m.buses, initialize=busOut_init)

        def busIn_init(m, bus):
            for i, j in m.arcs:
                if j == bus:
                    yield i
        m.busIn = Set(m.buses, initialize=busIn_init)
    
        m.varFlow = Var(m.t, m.arcs, domain=NonNegativeReals, initialize=0)
        def flow_limit(m, t, a, b):
            if network[(a,b)].power_lim is None:
                return Constraint.Skip
            return m.varFlow[t, a, b] <= network[(a,b)].power_lim
        m.eq_flow_limits = Constraint(m.t, m.arcs, rule=flow_limit) 

        def eq_flow_balance(m, t, bus):
            demand = load.get(bus, None)
            demand = 0 if demand is None else demand[t]
            return sum(m.varPower[t, u] for u in m.units if network.link(bus, u)) \
                + sum(m.varFlow[t, i, bus] for i in m.busIn[bus]) \
                - sum(m.varFlow[t, bus, j] for j in m.busOut[bus]) \
                == demand
        m.eq_flow_balance = Constraint(m.t, m.buses, rule=eq_flow_balance)
        
        r = network.system.reserve_req
        if 1 > r > 0:
            m.varFlowReserve = Var(m.t, m.arcs, domain=NonNegativeReals)
            # Power reserve
            def eq_flow_reserve(m, t, bus):
                return sum(m.varPower[t, u] for u in m.units if network.link(bus, u)) \
                    + sum(m.varFlowReserve[t, i, bus] for i in m.busIn[bus]) \
                    - sum(m.varFlowReserve[t, bus, j] for j in m.busOut[bus]) \
                    >= load[t, bus] *(1+r)
            m.eq_flow_reserve = Constraint(m.t, m.buses,
                rule=lambda m, t, bus: sum(m.varReserve[t,u] for u in m.units) >= load[bus][t] * (1+r)
                )
        self.m = m

    def _build_model(self, network, load):
        self.m   = ConcreteModel()
        T = len(list(load.values())[0])
        self.m.t = Set(initialize=list(range(T)), ordered=True)
        
        self._build_units_equations(network.system)
        self._build_reserve_equations(network.system)
        self._build_balance_equations(network, load)
        self._build_linear_objective(network.system)

        self.m.dual = Suffix(direction=Suffix.IMPORT)

    def get_lmp(self):
        # return a nested dictionary {bus : {t: dual}}
        return {b : {t: self.m.dual[self.m.eq_flow_balance[t,b]] for t in self.m.t} for b in self.m.buses}


class SCDCModel(DCModel):
    def __init__(self):
        super().__init__()

    def _build_security_constraints(self, network, load, contingencies='all'):
        m = self.m
        system = network.system

        if contingencies == 'all':
            m.contingencies = SetOf(m.arcs)
        else:
            m.contingencies = Set(within=m.arcs, initialize=contingencies)
        m.varPowerC = Var(m.t, m.units, m.contingencies, domain=NonNegativeReals)
        m.varFlowC = Var(m.t, m.arcs, m.contingencies, domain=NonNegativeReals, initialize=0)
            
        # Ramp Equations
        def eq_contingency_ramp_up(m, t, u, ca, cb):
            if  system[u].ramp_up is None or t == m.t.first():
                return Constraint.Skip
            else:
                return m.varPowerC[t, u, ca, cb] - m.varPowerC[t-1, u, ca, cb] <= system[u].ramp_up * m.binIsOn[t-1,u] + system[u].min_power * m.binStartUp[t,u]
        
        def eq_contingency_ramp_down(m, t, u, ca, cb):
            if  system[u].ramp_down is None or t == m.t.first():
                return Constraint.Skip
            else:
                return m.varPowerC[t-1, u, ca, cb] - m.varPowerC[t, u, ca, cb] <= system[u].ramp_down * m.binIsOn[t,u] + system[u].min_power * m.binShutDown[t,u]
        m.eq_contingency_ramp_up = Constraint(m.t, m.units, m.contingencies, rule=eq_contingency_ramp_up)
        m.eq_contingency_ramp_down = Constraint(m.t, m.units, m.contingencies, rule=eq_contingency_ramp_down)

        # Min Max Power
        def eq_contingency_min_power(m, t, u, ca, cb):
            return m.varPowerC[t, u, ca, cb] >= system[u].min_power * m.binIsOn[t, u]
        def eq_contingency_max_power(m, t, u, ca, cb):
            return m.varPowerC[t, u, ca, cb] <= system[u].max_power * m.binIsOn[t, u]
        m.eq_contingency_min_power = Constraint(m.t, m.units, m.contingencies, rule=eq_contingency_min_power)
        m.eq_contingency_max_power = Constraint(m.t, m.units, m.contingencies, rule=eq_contingency_max_power)
            
        # Raction Rate Equations
        def eq_contingency_react_up(m, t, u, ca, cb):
            if  system[u].ramp_up is None:
                return Constraint.Skip
            else:
                return m.varPowerC[t, u, ca, cb] - m.varPower[t, u] <= system[u].ramp_up/2
        def eq_contingency_react_down(m, t, u, ca, cb):
            if  system[u].ramp_down is None:
                return Constraint.Skip
            else:
                return m.varPower[t, u] - m.varPowerC[t, u, ca, cb] <= system[u].ramp_down/2
        m.eq_contingency_react_up = Constraint(m.t, m.units, m.contingencies, rule=eq_contingency_react_up)
        m.eq_contingency_react_down = Constraint(m.t, m.units, m.contingencies, rule=eq_contingency_react_down)

        def eq_contingency_flow_limit(m, t, a, b, ca, cb):
            if network[(a,b)].power_lim is None:
                return Constraint.Skip
            return m.varFlowC[t, a, b, ca, cb] <= network[(a,b)].power_lim
        m.eq_contingency_flow_limits = Constraint(m.t, m.arcs, m.contingencies, rule=eq_contingency_flow_limit)

        def eq_contingency_flow_balance(m, t, bus, ca, cb):
            demand = load.get(bus, None)
            demand = 0 if demand is None else demand[t]
            return sum(m.varPowerC[t, u, ca, cb] for u in m.units if network.link(bus, u)) \
                + sum(m.varFlowC[t, i, bus, ca, cb] for i in m.busIn[bus]) \
                - sum(m.varFlowC[t, bus, j, ca, cb] for j in m.busOut[bus]) \
                == demand
        m.eq_contingency_flow_balance = Constraint(m.t, m.buses, m.contingencies, rule=eq_contingency_flow_balance)

        # Fix flow C in lane at 0
        for c in m.contingencies:
            for t in m.t:
                m.varFlowC[t, c, c].fix(0)
        
        self.m = m
        
    def _build_model(self, network, load, contingencies):
        super()._build_model(network, load)
        self._build_security_constraints(network, load, contingencies)
        
    def solve(self, system, load, contingencies='all', tee=0, force_build=False):
        self._build_model(system, load, contingencies)    
        sol = SolverFactory("cbc", executable="./solvers/cbc/cbc.exe")
        sol.options["ratio"] = 0.01
        sol.options["sec"] = 200
        res = sol.solve(self.m, tee=tee)
        return value(self.m.cost)