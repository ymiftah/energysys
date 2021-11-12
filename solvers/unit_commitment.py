from .economic_dispatch import QPModel
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
                        m.varFuelCons[t, u]
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
        
    def solve(self, system, load, tee=0, exec="./solvers/cbc/cbc.exe"):
        self._build_model(system, load)    
        sol = SolverFactory("cbc", executable=exec)
        sol.options["ratio"] = 0.01
        sol.options["sec"] = 200
        res = sol.solve(self.m, tee=tee)
        status = (res.solver.status == SolverStatus.ok) and (res.solver.termination_condition == TerminationCondition.optimal)
        return status, value(self.m.cost)
    
    def get_power(self):
        data = ((key[0], key[1], val) for key, val in self.m.varPower.extract_values().items())
        df = pd.DataFrame(data, columns =['Time', 'Unit', 'Power'])
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
    

        m.varAngle = Var(m.t, m.buses, domain=Reals, initialize=0)
        for tt in m.t:
            m.varAngle[tt, m.buses.first()].fix(0)
        def flow_limit_up(m, t, a, b):
            if network.power_lim(a,b) is None:
                return Constraint.Skip
            return network.Z(a,b) * (m.varAngle[t, a] - m.varAngle[t, b]) <= network.power_lim(a,b)
        m.eq_flow_limits_up = Constraint(m.t, m.arcs, rule=flow_limit_up) 
        def flow_limit_lo(m, t, a, b):
            if network.power_lim(a,b) is None:
                return Constraint.Skip
            return network.Z(a,b) * (m.varAngle[t, a] - m.varAngle[t, b]) >= - network.power_lim(a,b)
        m.eq_flow_limits_lo = Constraint(m.t, m.arcs, rule=flow_limit_lo) 

        def eq_flow_balance(m, t, bus):
            demand = load.get(bus, None)
            demand = 0 if demand is None else demand[t]
            return sum(m.varPower[t, u] for u in m.units if network.link(bus, u)) \
                - demand \
                == sum(network.Z(bus, i) * (m.varAngle[t, bus] - m.varAngle[t, i])
                        for i in m.buses if i != bus)
        m.eq_flow_balance = Constraint(m.t, m.buses, rule=eq_flow_balance)
        
        r = network.system.reserve_req
        if 1 > r > 0:
            m.eq_flow_reserve = Constraint(m.t, m.buses,
                rule=lambda m, t, bus:
                    sum(m.varReserve[t,u] for u in m.units) \
                        >= load[bus][t] * (1+r)
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
        data = {b : {t: self.m.dual[self.m.eq_flow_balance[t,b]] for t in self.m.t} for b in self.m.buses}

        data = ((t, b, lmp) for b, val in data.items() for t, lmp in val.items())
        df = pd.DataFrame(data, columns =['Time', 'Node', 'LMP'])
        return df

    def get_lines_power(self, network):
        data = {(a,b) : {t: (value(self.m.varAngle[t, a]) - value(self.m.varAngle[t, b])) * network.Z(a,b)
                        for t in self.m.t} for a,b in self.m.arcs}
        data = ((t, a, b, power) for (a,b), val in data.items() for t, power in val.items())
        df = pd.DataFrame(data, columns =['Time', 'Node A', 'Node B', 'Power'])
        return df


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


        m.varAngleC = Var(m.t, m.buses, m.contingencies, domain=Reals, initialize=0)
        for c in m.contingencies:
            for tt in m.t:
                m.varAngleC[tt, m.buses.first(), m.contingencies].fix(0)
        def contingencies_flow_limit_up(m, t, a, b, ca, cb):
            if network.power_lim(a,b) is None:
                return Constraint.Skip
            return network.Z(a,b) * (m.varAngleC[t, a, ca, cb] - m.varAngleC[t, b, ca, cb]) <= network.power_lim(a,b)
        m.eq_contingencies_flow_limits_up = Constraint(m.t, m.arcs, m.contingencies, rule=contingencies_flow_limit_up) 
        def contingencies_flow_limit_lo(m, t, a, b, ca, cb):
            if network.power_lim(a,b) is None:
                return Constraint.Skip
            return network.Z(a,b) * (m.varAngleC[t, a, ca, cb] - m.varAngleC[t, b, ca, cb]) >= - network.power_lim(a,b)
        m.eq_contingencies_flow_limits_lo = Constraint(m.t, m.arcs, m.contingencies, rule=contingencies_flow_limit_lo) 


        def contingencies_eq_flow_balance(m, t, bus, ca, cb):
            demand = load.get(bus, None)
            demand = 0 if demand is None else demand[t]
            return sum(m.varPowerC[t, u, ca, cb] for u in m.units if network.link(bus, u)) \
                - demand \
                == sum(network.Z(bus, i) * (m.varAngleC[t, bus, ca, cb] - m.varAngleC[t, i, ca, cb])
                        for i in m.buses if i != bus)
        m.eq_contingencies_flow_balance = Constraint(m.t, m.buses, m.contingencies, rule=contingencies_eq_flow_balance)
    
        # Fix flow C in lane at 0
        for c in m.contingencies:
            for t in m.t:
                m.varFlowC[t, c, c].fix(0)
        
        self.m = m
        
    def _build_model(self, network, load, contingencies):
        super()._build_model(network, load)
        self._build_security_constraints(network, load, contingencies)
        
    def solve(self, system, load, contingencies='all', tee=0, exec="./solvers/cbc/cbc.exe"):
        self._build_model(system, load, contingencies)    
        sol = SolverFactory("cbc", executable=exec)
        sol.options["ratio"] = 0.01
        sol.options["sec"] = 200
        res = sol.solve(self.m, tee=tee)
        status = (res.solver.status == SolverStatus.ok) and (res.solver.termination_condition == TerminationCondition.optimal)
        return status, value(self.m.cost)