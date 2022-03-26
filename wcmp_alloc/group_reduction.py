import gurobipy as gp
import numpy as np
import time
from gurobipy import GRB
from math import gcd, sqrt, isclose
from functools import reduce
from itertools import chain

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
    while any(list(map(lambda x: not isclose(x % 1, 0), frac_list))):
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

def l1_norm_diff(G1, G2):
    '''
    Computes the L1 norm of difference between group G1 and G2. They must be of
    the same size.
    '''
    assert len(G1) == len(G2) and G1 and G2, 'G1 and G2 must be non-empty ' \
                                             'equal size.'
    G1, G2 = np.array(G1), np.array(G2)
    return np.linalg.norm((G2 / sum(G2) - G1 / sum(G1)), ord=1)

def input_groups_gen(g, p, lb, ub, frac_digits):
    '''
    Generate `g` input groups, each with `p` ports. The weight for each port is
    generated randomly from a uniform distribution between `lb` and `ub`. Each
    weight will be preserved with `frac_digits` in precision.

    Returns a list of lists.
    '''
    input_groups = np.random.uniform(lb, ub, size=(g, p)).tolist()
    return [[round(w, frac_digits) for w in g] for g in input_groups]

class GroupReduction:
    '''
    GroupReduction runs various solvers to reduce input groups to acceptable
    integer groups that can be directly implemented on switch hardware, based on
    defined problem formulation and constraints.
    '''
    def __init__(self, groups, traffic, table_limit=TABLE_LIMIT):
        '''
        groups: input groups of intended weights reflecting desired traffic
                distribution.
        traffic: input per group traffic volume in Gbps.

        orig_groups: a list of lists, where each element list is a set of
                     weights for the corresponding group.
        int_groups: original groups after lossless scaleup to integers.
        table_limit: upper bound of the group entries on the switch.
        '''
        self._orig_groups = groups
        self._int_groups = list(map(frac2int_lossless, groups))
        self._groups = self._int_groups if FLAG_USE_INT_INPUT_GROUPS else \
                                           self._orig_groups
        self._traffic = traffic
        self._table_limit = table_limit

    def solve_sssg(self, formulation='L1NORM2'):
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
            m.setParam("FeasibilityTol", 1e-7)
            m.setParam("IntFeasTol", 1e-8)
            m.setParam("MIPGap", 1e-4)
            m.setParam("LogToConsole", 1)
            #m.setParam("NodefileStart", 0.5)
            m.setParam("NodefileDir", "/tmp")
            m.setParam("Threads", 0)
            #m.setParam("TimeLimit", 100)
            #m.setParam("LogFile", "gurobi.log")

            # Construct model
            if formulation == "COSSIM1":
                m.setParam("NonConvex", 2)
                m.setParam("MIPFocus", 2)
                m = self._sssg_cosine_similarity_1(m)
            elif formulation == "COSSIM2":
                m.setParam("NonConvex", 2)
                m.setParam("MIPFocus", 2)
                m = self._sssg_cosine_similarity_2(m)
            elif formulation == "L1NORM1":
                m = self._sssg_l1_norm_1(m)
            elif formulation == "L1NORM2":
                m.setParam("NonConvex", 2)
                m.setParam("MIPFocus", 2)
                m = self._sssg_l1_norm_2(m)
            else:
                print("Formulation not recognized!")
                return []

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
        # Create variables: wf[i] is intended (fractional) weight, w[i] is
        # actual (integral) weight. ws[i] is the square of w[i].
        wf, w, ws = self._groups[0], [], []
        for n in range(len(wf)):
            w.append(m.addVar(vtype=GRB.INTEGER, lb=0,
                               ub=self._table_limit,
                               name="w_" + str(n+1)))
            ws.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                ub=self._table_limit * self._table_limit,
                                name="ws_" + str(n+1)))
        z = m.addVar(vtype=GRB.CONTINUOUS, name="z")
        zs = m.addVar(vtype=GRB.CONTINUOUS, name="zs")

        # Objective is quadratic.
        obj = gp.QuadExpr();
        # fastest way to construct a large objective.
        # Params are: coeffs, var1s, var2s (must be of same size).
        obj.addTerms(wf, w, [z] * len(wf))
        # Set objective
        m.setObjective(obj, GRB.MAXIMIZE)

        # Add constraint: sum(w) <= table_limit
        m.addConstr(gp.quicksum(w) <= self._table_limit, "group_size_ub")
        # Add constraint: sum(w) >= 1 (group cannot be empty)
        m.addConstr(gp.quicksum(w) >= 1, "group_size_lb")
        # Add constraint: zs * sum(wf^2) * sum(wis) == 1
        c = gp.QuadExpr()
        c.addTerms([sum(v*v for v in wf)] * len(ws), [zs] * len(ws), ws)
        m.addConstr(c == 1, "linearization_zs")
        # Add constraint: zs = z * z
        m.addConstr(zs == z * z, "linearization_z")
        for i in range(len(w)):
            # Add constraint: ws = w * w
            m.addConstr(ws[i] == w[i] * w[i],
                        "linearization_ws_" + str(1 + i))
            # Add constraint only if inputs are already scaled up to integers:
            # w[i] <= wf[i]
            if FLAG_USE_INT_INPUT_GROUPS:
                m.addConstr(w[i] <= wf[i], "no_scale_up_" + str(1+i))

        return m

    def _sssg_cosine_similarity_2(self, m):
        '''
        Build a non-convex MINLP formulation using cosine similarity as
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
        m.update()

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

    def _sssg_l1_norm_1(self, m):
        '''
        Build an MILP formulation using L1 norm as the objective for the single
        switch single group (SSSG) optimization. In L1 norm, actual weights to
        be solved are normalized against the swtich table limit.

        m: pre-built empty model, needs decision vars and constraints.
        '''
        # Create variables: wf[i] is the intended (fractional) weight after
        # normalization, w[i] is the actual (integral) weight.
        wf, wf_sum, w, u = self._groups[0], sum(self._groups[0]), [], []
        for n in range(len(wf)):
            w.append(m.addVar(vtype=GRB.INTEGER, lb=0, ub=self._table_limit,
                              name="w_" + str(n+1)))
            u.append(m.addVar(vtype=GRB.CONTINUOUS, name="u_" + str(n+1)))
        l1_norm = m.addVar(vtype=GRB.CONTINUOUS, name="l1_norm")

        # Set objective
        m.setObjective(l1_norm, GRB.MINIMIZE)

        # Force l1_norm to be sum(u), which is the sum of all absolute values.
        m.addConstr(l1_norm == gp.quicksum(u))
        # Add constraint: sum(w) <= table_limit
        m.addConstr(gp.quicksum(w) <= self._table_limit, "group_size_ub")
        for i in range(len(w)):
            # Add constraint: u[i] == abs(w[i] / table_limit - wf[i]).
            m.addConstr(w[i] / self._table_limit - wf[i] / wf_sum <= u[i])
            m.addConstr(wf[i] / wf_sum - w[i] / self._table_limit <= u[i])
            # Add constraint only if inputs are already scaled up to integers:
            # w[i] <= wf[i]
            if FLAG_USE_INT_INPUT_GROUPS:
                m.addConstr(w[i] <= wf[i], "no_scale_up_" + str(1+i))

        return m

    def _sssg_l1_norm_2(self, m):
        '''
        Build an MIQCP formulation using L1 norm as the objective for the single
        switch single group (SSSG) optimization. In L1 norm, actual weights to
        be solved are normalized against the sum of actual weights.

        m: pre-built empty model, needs decision vars and constraints.
        '''
        # Create variables: wf[i] is the intended (fractional) weight after
        # normalization, w[i] is the actual (integral) weight.
        wf, wf_sum, w, u = self._groups[0], sum(self._groups[0]), [], []
        for n in range(len(wf)):
            w.append(m.addVar(vtype=GRB.INTEGER, lb=0, ub=self._table_limit,
                              name="w_" + str(n+1)))
            u.append(m.addVar(vtype=GRB.CONTINUOUS, name="u_" + str(n+1)))
        z = m.addVar(vtype=GRB.CONTINUOUS, name="z")
        l1_norm = m.addVar(vtype=GRB.CONTINUOUS, name="l1_norm")

        # Set objective
        m.setObjective(l1_norm, GRB.MINIMIZE)

        # Force l1_norm to be sum(u), which is the sum of all absolute values.
        m.addConstr(l1_norm == gp.quicksum(u))
        # Add constraint: sum(w) <= table_limit.
        m.addConstr(gp.quicksum(w) <= self._table_limit, "group_size_ub")
        # Add constraint: z == 1/sum(w).
        m.addConstr(z * gp.quicksum(w) == 1, "z_limitation")
        for i in range(len(w)):
            # Add constraint: u[i] == abs(w[i] / table_limit - wf[i]).
            m.addConstr(w[i] * z - wf[i] / wf_sum <= u[i])
            m.addConstr(wf[i] / wf_sum - w[i] * z <= u[i])
            # Add constraint only if inputs are already scaled up to integers:
            # w[i] <= wf[i]
            if FLAG_USE_INT_INPUT_GROUPS:
                m.addConstr(w[i] <= wf[i], "no_scale_up_" + str(1+i))

        return m

    def solve_ssmg(self, formulation='L1NORM'):
        '''
        Given the input groups and table limit, solve the single switch multi 
        group (SSMG) optimization problem.
        '''
        if len(self._groups) <= 0:
            print('[ERROR] %s: unexpected number of input groups %s' %
                  solve_ssmg.__name__, len(self._groups))
            return []
        if len(self._groups) != len(self._traffic):
            print('[ERROR] %s: group size %s and traffic size %s mismatch' %
                  solve_sssg.__name__, len(self._groups), len(self._traffic))
            return []
        final_groups = self._groups.copy()

        try:
            # Initialize a new model
            m = gp.Model("single_switch_multi_group")
            m.setParam("FeasibilityTol", 1e-7)
            m.setParam("IntFeasTol", 1e-8)
            m.setParam("MIPGap", 1e-4)
            m.setParam("LogToConsole", 1)
            #m.setParam("NodefileStart", 0.5)
            m.setParam("NodefileDir", "/tmp")
            m.setParam("Threads", 0)
            #m.setParam("TimeLimit", 100)
            #m.setParam("LogFile", "gurobi.log")

            # Construct model
            if formulation == "L1NORM":
                m.setParam("NonConvex", 2)
                m.setParam("MIPFocus", 2)
                m = self._ssmg_l1_norm(m)
            else:
                print("Formulation not recognized!")
                return []

            # Optimize model
            m.optimize()

            sol_w = dict()
            for v in m.getVars():
                if 'w_' in v.VarName:
                    split = v.VarName.split('_')
                    final_groups[int(split[1])-1][int(split[2])-1] = round(v.X)
                    sol_w[v.VarName] = v.X
            print('Obj: %s' % m.ObjVal)
            print(*sol_w.items(), sep='\n')
            print('wf: %s' % self._groups)
            # Applies a final GCD reduction just in case.
            final_groups = list(map(frac2int_lossless, final_groups))

            return final_groups

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
            return []
        except AttributeError:
            print('Encountered an attribute error')
            return []

    def _ssmg_l1_norm(self, model):
        '''
        Build an ILP formulation using L1 norm as the objective for the single
        switch multi group (SSMG) optimization.

        model: pre-built empty model, needs decision vars and constraints.
        '''
        # Create variables: wf[m][i] is the intended (fractional) weight for
        # port i of group m after normalization, w[i] is the actual (integral)
        # weight.
        C, T = self._table_limit, self._traffic
        wf, wf_sum = self._groups, [sum(g) for g in self._groups]
        w, u, l1_norm, z = [], [], [], []
        for m in range(len(wf)):
            wm, um = [], []
            for i in range(len(wf[m])):
                wm.append(model.addVar(vtype=GRB.INTEGER, lb=0, ub=C,
                                       name="w_{}_{}".format(m+1, i+1)))
                um.append(model.addVar(vtype=GRB.CONTINUOUS,
                                       name="u_{}_{}".format(m+1, i+1)))
            w.append(wm)
            u.append(um)
            l1_norm.append(model.addVar(vtype=GRB.CONTINUOUS,
                                        name="l1_norm_{}".format(m+1)))
            z.append(model.addVar(vtype=GRB.CONTINUOUS,
                                  name="z_{}".format(m+1)))

        # Set objective: sum(T[m] * l1_norm[m]).
        model.setObjective(gp.LinExpr(T, l1_norm), GRB.MINIMIZE)

        # Add constraint: sum(w[m][i]) <= table_limit. Note that w is flattened
        # from a 2D list to 1D.
        model.addConstr(gp.quicksum(list(chain.from_iterable(w))) <= C,
                        "group_size_ub")
        for m in range(len(wf)):
            # Add constraint: per-group L1 norm.
            model.addConstr(l1_norm[m] == gp.quicksum(u[m]))
            # Add constraint: z[m] = 1 / sum(w[m])
            model.addConstr(z[m] * gp.quicksum(w[m]) == 1,
                            "z_{}_limitation".format(m+1))
            for i in range(len(wf[m])):
                # Add constraint: u[m][i] == abs(w[m][i] / C - wf[m][i]).
                model.addConstr(w[m][i] * z[m] - wf[m][i] / wf_sum[m] <= u[m][i])
                model.addConstr(wf[m][i] / wf_sum[m] - w[m][i] * z[m] <= u[m][i])
                # Add constraint only if inputs are already scaled up to
                # integers: w[m][i] <= wf[m][i]
                if FLAG_USE_INT_INPUT_GROUPS:
                    model.addConstr(w[m][i] <= wf[m][i],
                                    "no_scale_up_{}_{}".format(m+1, i+1))

        return model

if __name__ == "__main__":
    table_limit = 16*1024
    # groups, # port per group, lower bound, upper bound, fraction precision
    g, p, lb, ub, frac_digits = 2, 2, 1, 100, 3
    # Assuming uniform unit traffic.
    traffic_vol = [1] * g

    #input_groups = [[1.1, 2.1], [3.1, 4.1]]
    input_groups = input_groups_gen(g, p, lb, ub, frac_digits)
    start = time.time_ns()
    group_reduction = GroupReduction(input_groups, traffic_vol, table_limit)
    output_groups = group_reduction.solve_ssmg('L1NORM')
    end = time.time_ns()
    print('Input: %s' % input_groups)
    print('Output: %s' % output_groups)
    for i in range(len(input_groups)):
        diff = l1_norm_diff(input_groups[i], output_groups[i])
        print('Group {} L1 Norm: {}'.format(i, diff))
    print('Solving time (msec):', (end - start)/10**6)
