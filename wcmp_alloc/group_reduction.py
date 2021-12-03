import gurobipy as gp
from gurobipy import GRB
import math

# Broadcom Tomahawk 2 ECMP table limit.
TABLE_LIMIT = 16 * 1024

class GroupReduction:
    '''
    GroupReduction runs various solvers to reduce input groups to acceptable
    integer groups that can be directly implemented on switch hardware, based on
    defined problem formulation and constraints.
    '''
    def __init__(self, groups, table_limit=TABLE_LIMIT):
        '''
        groups: a list of lists, where each element list is a set of weights for
                the corresponding group.
        table_limit: upper bound of the group entries on the switch.
        '''
        self._groups = groups
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
            m.params.NonConvex = 2

            # Create variables: wf[i] is intended (fractional) weight, wi[i] is
            # actual (integral) weight. wis[i] is the square of wi[i].
            wf, wi, wis = self._groups[0], [], []
            for n in range(len(wf)):
                wi.append(m.addVar(vtype=GRB.INTEGER, name="wi_" + str(n+1)))
                wis.append(m.addVar(vtype=GRB.INTEGER, name="wis_" + str(n+1)))
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
            m.addConstr(gp.quicksum(wi) <= self._table_limit, "table_limit")
            # Add constraint: sum(wi) > 0 (constraint needs to be binding, vars
            # cannot be 0, so this is a workaround)
            m.addConstr(gp.quicksum(wi) >= 0.000001, "non_empty_group")
            # Add constraint: zs * sum(wf^2) * sum(wis) == 1
            c2 = gp.QuadExpr()
            c2.addTerms([sum(v*v for v in wf)] * len(wis), [zs] * len(wis), wis)
            m.addConstr(c2 == 1, "linearization_zs")
            # Add constraint: zs = z * z
            m.addConstr(zs == z * z, "linearization_z")
            for i in range(len(wi)):
                # Add constraint: wis = wi * wi
                m.addConstr(wis[i] == wi[i] * wi[i],
                            "linearization_wis_" + str(1 + i))
                # Add constraint: wi[i] <= ceil(wf[i])
                m.addConstr(wi[i] <= int(math.ceil(wf[i])), "no_scaling_up")

            # Optimize model
            m.optimize()

            group = []
            for v in m.getVars():
                if 'wi_' in v.VarName:
                    group.append(v.X)
                    print('%s %g' % (v.VarName, v.X))
            print('Obj: %g' % m.ObjVal)
            final_groups.append(group)

            return final_groups

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
            return []
        except AttributeError:
            print('Encountered an attribute error')
            return []
