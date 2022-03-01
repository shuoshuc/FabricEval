import gurobipy as gp
from gurobipy import GRB
from math import gcd, sqrt
from functools import reduce

# If True, feeds solver with scaled up integer groups.
FLAG_USE_INT_INPUT_GROUPS = True

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

def cosine_similarity(G1, G2):
    '''
    Computes the cosine similarity between group G1 and G2. They must be of the
    same size.
    '''
    assert len(G1) == len(G2) and G1 and G2, 'G1 and G2 must be non-empty ' \
                                             'equal size.'
    dot_sum, g1_l2_norm_sq, g2_l2_norm_sq = 0, 0, 0
    for i in range(len(G1)):
        dot_sum += G1[i] * G2[i]
        g1_l2_norm_sq += G1[i] * G1[i]
        g2_l2_norm_sq += G2[i] * G2[i]
    return dot_sum / (sqrt(g1_l2_norm_sq) * sqrt(g2_l2_norm_sq))

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
            # Initialize a new model
            m = gp.Model("single_switch_single_group")
            m.setParam("NonConvex", 2)
            m.setParam("FeasibilityTol", 1e-9)
            m.setParam("IntFeasTol", 1e-9)
            m.setParam("MIPGap", 1e-9)
            m.setParam("LogToConsole", 1)
            m.setParam("NodefileStart", 10)
            m.setParam("NodefileDir", "/tmp")
            m.setParam("Threads", 0)
            #m.setParam("LogFile", "gurobi.log")

            # Construct model
            m = self._sssg_cosine_similarity_2(m)
            # Optimize model
            m.optimize()

            group = []
            sol_w, sol_z = dict(), dict()
            for v in m.getVars():
                if 'w_' in v.VarName:
                    group.append(round(v.X))
                    sol_w[v.VarName] = v.X
                if 'z_' in v.VarName:
                    sol_z[v.VarName] = v.X
            print('Obj: %s' % m.ObjVal)
            print('Cosine similarity: %s' % \
                  cosine_similarity(self._groups[0], list(sol_w.values())))
            print(*sol_w.items(), sep='\n')
            print(*sol_z.items(), sep='\n')
            print('wf: %s' % self._groups[0])
            # Applies a final GCD reduction just in case.
            final_groups.append(frac2int_lossless(group))

            return final_groups

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
            return []
        except AttributeError:
            print('Encountered an attribute error')
            return []

    def _sssg_cosine_similarity_1(self, m):
        '''
        Build a non-convex MINLP (QP+QC) formulation using cosine similarity as
        objective for the single switch single group (SSSG) optimization.

        m: pre-built empty model, needs decision vars and constraints.
        '''
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

        return m

    def _sssg_cosine_similarity_2(self, m):
        '''
        Build a non-convex MIQCP formulation using cosine similarity as
        objective for the single switch single group (SSSG) optimization.

        m: pre-built empty model, needs decision vars and constraints.
        '''
        # Create variables: wf[i] is intended (fractional) weight, w[i] is
        # actual (integral) weight.
        wf, w, z, ws, zs = self._groups[0], [], [], [], []
        for n in range(len(wf)):
            w.append(m.addVar(vtype=GRB.INTEGER, lb=0, ub=self._table_limit,
                              name="w_" + str(n+1)))
            w[n].start = self._int_groups[0][n]
            ws.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0,
                               ub=self._table_limit * self._table_limit,
                               name="ws_" + str(n+1)))
            ws[n].start = self._int_groups[0][n] * self._int_groups[0][n]
            z.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                              name="z_" + str(n+1)))
            zs.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                               name="zs_" + str(n+1)))

        # Objective is linear.
        obj = gp.LinExpr(wf, z);
        # Set objective
        m.setObjective(1/sqrt(sum(v*v for v in wf)) * obj, GRB.MAXIMIZE)

        # Add constraint: sum(wi) <= table_limit
        m.addConstr(gp.quicksum(w) <= self._table_limit, "group_size_ub")
        # Add constraint: sum(wi) >= 1 (group cannot be empty)
        m.addConstr(gp.quicksum(w) >= 1, "group_size_lb")
        for i in range(len(w)):
            # Add constraint: zs[i] * sum(ws) == ws[i]
            c = gp.QuadExpr()
            c.addTerms([1] * len(ws), [zs[i]] * len(ws), ws)
            m.addConstr(c == ws[i], "binding_w_z_" + str(1+i))
            # Add constraint: zs[i] = z[i] * z[i]
            m.addConstr(zs[i] == z[i] * z[i], "linearization_z_" + str(1+i))
            # Add constraint: ws[i] = w[i] * w[i]
            m.addConstr(ws[i] == w[i] * w[i], "linearization_w_" + str(1+i))
            # Add constraint only if inputs are already scaled up to integers:
            # w[i] <= wf[i]
            if FLAG_USE_INT_INPUT_GROUPS:
                m.addConstr(w[i] <= wf[i], "no_scale_up_" + str(1+i))

        return m

if __name__ == "__main__":
    #input_groups = [[10.5, 20.1, 31.0, 39.7]]
    #input_groups = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    input_groups = [[1.1, 2.1, 3.1, 4.1]]
    #input_groups = [[i for i in range(1, 17)]]
    group_reduction = GroupReduction(input_groups, 16*1024)
    output_groups = group_reduction.solve_sssg()
    print('Input: %s' % input_groups)
    print('Output: %s' % output_groups)
    res = cosine_similarity(input_groups[0], output_groups[0])
    print('Input/Output cosine similarity: %s' % res)
