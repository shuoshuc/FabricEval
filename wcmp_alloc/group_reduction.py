import gurobipy as gp
import numpy as np
import time
import copy
from gurobipy import GRB
from math import gcd, sqrt, isclose, ceil, floor
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

def calc_group_oversub(G, G_prime, mode='max'):
    '''
    Group oversub delta(G, G') is defined as the max port oversub of the
    group, as per equation 2 in the EuroSys WCMP paper. Used in the group
    reduction algoritm (Algorithm 1 in the paper).

    G: original group before reduction.
    G': final group after reduction.
    mode: a knob that allows the function to use avg or percentile instead
          of max to find the group oversub. Must be one of 'max', 'avg', 'p50'.
    Return: group oversub.
    '''
    G_sum = sum(G)
    G_prime_sum = sum(G_prime)
    numerator = np.asarray(G_prime) * G_sum
    demominator = np.asarray(G) * G_prime_sum

    if mode == 'max':
        return np.max(numerator / demominator)
    elif mode == 'avg':
        return np.average(numerator / demominator)
    elif mode == 'p50':
        return np.percentile(numerator / demominator, 50)
    else:
        print('Mode %s not recognized!' % mode)
        return None


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
        table_limit: upper bound of the group entries on the switch. Note that
                     this does not have to be the physical limit, but can also
                     be the available headroom left.
        '''
        self._orig_groups = copy.deepcopy(groups)
        self._int_groups = list(map(frac2int_lossless, copy.deepcopy(groups)))
        self._groups = self._int_groups if FLAG_USE_INT_INPUT_GROUPS else \
                                           self._orig_groups
        self._traffic = traffic
        self._table_limit = table_limit

    def _choose_port_to_update(self, group_to_reduce, group_under_reduction):
        '''
        Helper function for the reduce_wcmp_group algorithm.
        Iteratively goes through all ports of the group under reduction,
        returns the index of member port whose weight should be incremented to
        result in least maximum oversub.
        '''
        if len(group_to_reduce) != len(group_under_reduction):
            print('[ERROR] %s: group dimension mismatch %s %s' %
                  GroupReduction._choose_port_to_update.__name__,
                  len(group_to_reduce), len(group_under_reduction))
            return -1

        min_oversub, index = float('inf'), -1
        gtr_size = sum(group_to_reduce)
        gur_size = sum(group_under_reduction)
        for i in range(len(group_under_reduction)):
            oversub = ((group_under_reduction[i] + 1) *
                       gtr_size) / ((gur_size + 1) * group_to_reduce[i])
            if min_oversub > oversub:
                min_oversub = oversub
                index = i

        return index

    def reduce_wcmp_group(self, group_to_reduce, delta_max=1.2):
        '''
        Single group weight reduction algoritm with an oversub limit.
        This is algorithm 1 in the EuroSys WCMP paper.
        '''
        final_group = np.ones(len(group_to_reduce))
        while calc_group_oversub(group_to_reduce, final_group) > delta_max:
            index = self._choose_port_to_update(group_to_reduce, final_group)
            final_group[index] += 1
            if sum(final_group) >= sum(group_to_reduce):
                print('I give up!!')
                return group_to_reduce

        return final_group.tolist()

    def table_fitting_sssg(self):
        '''
        WCMP weight reduction for table fitting a single WCMP group into table
        limit. Assuming that we are fitting self._int_groups, since
        self._orig_groups do not necessarily have integer weights.
        '''
        if len(self._int_groups) != 1:
            print('[ERROR] %s: unexpected number of input groups %s' %
                  GroupReduction.table_fitting_sssg.__name__,
                  len(self._int_groups))
            return []

        group_to_reduce = self._int_groups[0]
        T = self._table_limit
        final_group = copy.deepcopy(group_to_reduce)
        while sum(final_group) > T:
            non_reducible_size = 0
            # Counts singleton ports, as they cannot be reduced any further.
            for i in range(len(final_group)):
                if final_group[i] == 1:
                    non_reducible_size += 1
            # If none of the port weights can be reduced further, just give up.
            if non_reducible_size == len(final_group):
                break
            # Directly shrinks original weights by `reduction_ratio` so that
            # the final group can fit into T. Note that the denominator should
            # technically be sum(group_to_reduce) - non_reducible_size since the
            # singleton ports should just be left out of reduction. But the
            # algorithm still reduces (to 0) and then always rounds up to 1.
            reduction_ratio = (T - non_reducible_size) / sum(group_to_reduce)
            for i in range(len(final_group)):
                final_group[i] = np.floor(group_to_reduce[i] * reduction_ratio)
                if final_group[i] == 0:
                    final_group[i] = 1

        # In case the previous round-down over-reduced groups, which leaves some
        # headroom in T, we make full use of the headroom while minimizing the
        # oversub.
        remaining_size = int(T - sum(final_group))
        min_oversub = calc_group_oversub(group_to_reduce, final_group)
        final_group_2 = copy.deepcopy(final_group)
        for _ in range(remaining_size):
            index = self._choose_port_to_update(group_to_reduce, final_group)
            final_group[index] += 1
            curr_oversub = calc_group_oversub(group_to_reduce, final_group)
            if min_oversub > curr_oversub:
                final_group_2 = copy.deepcopy(final_group)
                min_oversub = curr_oversub

        return [final_group_2]

    def table_fitting_ssmg(self):
        '''
        WCMP weight reduction for table fitting a set of WCMP groups H into
        size S. Algorithm 4 in the EuroSys WCMP paper.
        '''
        if len(self._int_groups) <= 0:
            print('[ERROR] %s: unexpected number of input groups %s' %
                  GroupReduction.table_fitting_ssmg.__name__,
                  len(self._int_groups))
            return []

        enforced_oversub = 1.002
        step_size = 0.001
        S = self._table_limit
        # Sort groups in descending order of size.
        groups_in = sorted(self._int_groups, key=sum, reverse=True)
        groups_out = copy.deepcopy(groups_in)
        total_size = sum([sum(g) for g in groups_in])

        while total_size > S:
            for i in range(len(groups_in)):
                groups_out[i] = self.reduce_wcmp_group(groups_in[i],
                                                       enforced_oversub)
                total_size = sum([sum(g) for g in groups_out])
                if total_size <= S:
                    return groups_out
            # Relaxes oversub limit if we fail to fit all groups with the same
            # oversub.
            enforced_oversub += step_size

        return groups_out

    def solve_sssg(self, formulation='L1NORM3'):
        '''
        Given the input groups and table limit, solve the single switch single
        group (SSSG) optimization problem.
        '''
        final_groups = []
        if len(self._groups) != 1:
            print('[ERROR] %s: unexpected number of input groups %s' %
                  GroupReduction.solve_sssg.__name__, len(self._groups))
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
            # L1NORM1 normalizes actual integer weights over table limit.
            # L1NORM2 normalizes actual integer weights over its own group sum.
            # L1NORM3 minimizes L1 norm of absolute weights (no normalization).
            if formulation == "L1NORM1":
                m = self._sssg_l1_norm_1(m)
            elif formulation == "L1NORM2":
                m.setParam("NonConvex", 2)
                m.setParam("MIPFocus", 2)
                m = self._sssg_l1_norm_2(m)
            elif formulation == "L1NORM3":
                m.setParam("NonConvex", 2)
                m.setParam("MIPFocus", 2)
                m = self._sssg_l1_norm_3(m)
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

    def _sssg_l1_norm_1(self, m):
        '''
        Build an MILP formulation using L1 norm as the objective for the single
        switch single group (SSSG) optimization. In L1 norm, actual weights to
        be solved are normalized against the swtich table limit.

        m: pre-built empty model, needs decision vars and constraints.
        '''
        # Create variables: wf[i] is the intended (fractional) weight after
        # normalization, w[i] is the actual (integral) weight.
        wf, wf_sum, w, u = self._groups[0].copy(), sum(self._groups[0]), [], []
        for n in range(len(wf)):
            w.append(m.addVar(vtype=GRB.INTEGER, lb=0, ub=self._table_limit,
                              name="w_" + str(n+1)))
            u.append(m.addVar(vtype=GRB.CONTINUOUS, name="u_" + str(n+1)))

        # Set objective
        m.setObjective(gp.quicksum(u), GRB.MINIMIZE)

        # Add constraint: sum(w) <= table_limit
        m.addConstr(gp.quicksum(w) <= self._table_limit, "group_size_ub")
        for i in range(len(w)):
            # Add constraint: u[i] >= abs(w[i] / table_limit - wf[i]).
            # Note: '==' can be relaxed to '>=' because the objective is to
            # minimize sum(u[i]).
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
        wf, wf_sum, w, u = self._groups[0].copy(), sum(self._groups[0]), [], []
        for n in range(len(wf)):
            w.append(m.addVar(vtype=GRB.INTEGER, lb=0, ub=self._table_limit,
                              name="w_" + str(n+1)))
            u.append(m.addVar(vtype=GRB.CONTINUOUS, name="u_" + str(n+1)))
        z = m.addVar(vtype=GRB.CONTINUOUS, name="z")

        # Set objective
        m.setObjective(gp.quicksum(u), GRB.MINIMIZE)

        # Add constraint: sum(w) <= table_limit.
        m.addConstr(gp.quicksum(w) <= self._table_limit, "group_size_ub")
        # Add constraint: z == 1/sum(w).
        m.addConstr(z * gp.quicksum(w) == 1, "z_limitation")
        for i in range(len(w)):
            # Add constraint: u[i] >= abs(w[i] / sum(w[i]) - wf[i] / wf_sum).
            # Note: '==' can be relaxed to '>=' because the objective is to
            # minimize sum(u[i]).
            m.addConstr(w[i] * z - wf[i] / wf_sum <= u[i])
            m.addConstr(wf[i] / wf_sum - w[i] * z <= u[i])
            # Add constraint only if inputs are already scaled up to integers:
            # w[i] <= wf[i]
            if FLAG_USE_INT_INPUT_GROUPS:
                m.addConstr(w[i] <= wf[i], "no_scale_up_" + str(1+i))

        return m

    def _sssg_l1_norm_3(self, m):
        '''
        Build an MILP formulation using L1 norm as the objective for the single
        switch single group (SSSG) optimization. The L1 norm directly measures
        the absolute weight difference without normalization.

        m: pre-built empty model, needs decision vars and constraints.
        '''
        # Create variables: wf[i] is the intended (fractional) weight after
        # normalization, w[i] is the actual (integral) weight.
        wf, wf_sum, w, u = self._groups[0].copy(), sum(self._groups[0]), [], []
        for n in range(len(wf)):
            w.append(m.addVar(vtype=GRB.INTEGER, lb=0, ub=self._table_limit,
                              name="w_" + str(n+1)))
            u.append(m.addVar(vtype=GRB.CONTINUOUS, name="u_" + str(n+1)))

        # Set objective
        m.setObjective(gp.quicksum(u), GRB.MINIMIZE)

        # Add constraint: sum(w) <= table_limit.
        m.addConstr(gp.quicksum(w) <= self._table_limit, "group_size_ub")
        for i in range(len(w)):
            # Add constraint: u[i] >= abs(w[i] - wf[i]).
            # Note: '==' can be relaxed to '>=' because the objective is to
            # minimize sum(u[i]).
            m.addConstr(w[i] - wf[i] <= u[i])
            m.addConstr(wf[i] - w[i] <= u[i])
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
                  GroupReduction.solve_ssmg.__name__, len(self._groups))
            return []
        if len(self._groups) != len(self._traffic):
            print('[ERROR] %s: group size %s and traffic size %s mismatch' %
                  GroupReduction.solve_sssg.__name__, len(self._groups),
                  len(self._traffic))
            return []
        final_groups = copy.deepcopy(self._groups)

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
        wf, wf_sum = copy.deepcopy(self._groups), [sum(g) for g in self._groups]
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
