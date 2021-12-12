import gurobipy as gp
from gurobipy import GRB
from math import gcd
from functools import reduce

# If True, feeds solver with scaled up integer groups.
FLAG_USE_INT_INPUT_GROUPS = False

# Broadcom Tomahawk 2 ECMP table limit.
TABLE_LIMIT = 16 * 1024

def gcd_reduce(vector):
    '''
    Reduces a vector by its gcd.
    vector: input vector must be integers.
    '''
    return list(map(lambda x: x // reduce(gcd, vector), vector))

def frac2int_lossless(frac_list):
    '''
    Converts list of fractions to list of integers without loss. Essentially
    scaling up by multiplying 10s and then divide by gcd.
    '''
    while any(list(map(lambda x: x % 1, frac_list))):
        frac_list = list(map(lambda x: x * 10, frac_list))
    return gcd_reduce(list(map(int, frac_list)))

class GroupReduction:
    '''
    GroupReduction runs various solvers to reduce input groups to acceptable
    integer groups that can be directly implemented on switch hardware, based on
    defined problem formulation and constraints.
    '''
    def __init__(self, groups, table_limit=TABLE_LIMIT):
        '''
        orig_groups: a list of lists, where each element list is a set of
                     weights for the corresponding group.
        int_groups: original groups after lossless scaleup to integers.
        table_limit: upper bound of the group entries on the switch.
        '''
        self._orig_groups = groups
        self._int_groups = list(map(frac2int_lossless, groups))
        self._groups = self._int_groups if FLAG_USE_INT_INPUT_GROUPS else \
                                           self._orig_groups
        self._table_limit = table_limit

    def solve_sssg(self):
        '''
        Given the input groups and table limit, solve the single switch single
        group (SSSG) optimization problem.
        '''
        final_groups = []
        if len(self._groups) != 1:
            print('[ERROR] %s: unexpected number of input groups %s' %
                  solve_sssg.__name__, len(self._groups))
            return []

        try:
            # Create a new model
            m = gp.Model("single_switch_single_group")
            m.setParam("NonConvex", 2)
            m.setParam("FeasibilityTol", 1e-9)
            m.setParam("IntFeasTol", 1e-5)
            m.setParam("MIPGap", 1e-9)
            m.setParam("LogToConsole", 1)
            m.setParam("LogFile", "gurobi.log")

            # Create variables: wf[i] is intended (fractional) weight, wi[i] is
            # actual (integral) weight. wis[i] is the square of wi[i].
            wf, wi, wis = self._groups[0], [], []
            for n in range(len(wf)):
                wi.append(m.addVar(vtype=GRB.INTEGER, lb=0,
                                   ub=self._table_limit,
                                   name="wi_" + str(n+1)))
                wis.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                    ub=self._table_limit * self._table_limit,
                                    name="wis_" + str(n+1)))
            z = m.addVar(vtype=GRB.CONTINUOUS, name="z")
            zs = m.addVar(vtype=GRB.CONTINUOUS, name="zs")

            # Objective is quadratic.
            obj = gp.QuadExpr();
            # fastest way to construct a large objective.
            # Params are: coeffs, var1s, var2s (must be of same size).
            obj.addTerms(wf, wi, [z] * len(wf))
            # Set objective
            m.setObjective(obj, GRB.MAXIMIZE)

            # Add constraint: sum(wi) <= table_limit
            m.addConstr(gp.quicksum(wi) <= self._table_limit, "group_size_ub")
            # Add constraint: sum(wi) >= 1 (group cannot be empty)
            m.addConstr(gp.quicksum(wi) >= 1, "group_size_lb")
            # Add constraint: zs * sum(wf^2) * sum(wis) == 1
            c = gp.QuadExpr()
            c.addTerms([sum(v*v for v in wf)] * len(wis), [zs] * len(wis), wis)
            m.addConstr(c == 1, "linearization_zs")
            # Add constraint: zs = z * z
            m.addConstr(zs == z * z, "linearization_z")
            for i in range(len(wi)):
                # Add constraint: wis = wi * wi
                m.addConstr(wis[i] == wi[i] * wi[i],
                            "linearization_wis_" + str(1 + i))
            # Add constraint: 0 <= obj <= 1
            m.addConstr(obj <= 1, "cosine_similarity")
            m.addConstr(obj >= 0, "cosine_similarity")

            # Optimize model
            m.optimize()

            group = []
            for v in m.getVars():
                if 'wi_' in v.VarName:
                    group.append(round(v.X))
                    print('%s %s' % (v.VarName, v.X))
            print('Obj: %s' % m.ObjVal)
            # Applies a final GCD reduction just in case.
            final_groups.append(frac2int_lossless(group))

            return final_groups

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
            return []
        except AttributeError:
            print('Encountered an attribute error')
            return []


if __name__ == "__main__":
    input_groups = [[10.5, 20.1, 31.0, 39.7]]
    group_reduction = GroupReduction(input_groups, 16*1024)
    output_groups = group_reduction.solve_sssg()
    print('Input: %s' % input_groups)
    print('Output: %s' % output_groups)
