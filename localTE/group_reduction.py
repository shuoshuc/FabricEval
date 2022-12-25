import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from itertools import chain
from math import ceil, floor, gcd, isclose, sqrt

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from common.common import PRINTV
from common.flags import *


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

def frac2int_round(frac_list):
    '''
    Converts list of fractions to list of integers by simply rounding to the
    nearest integer. Exception is when the weight rounds to 0, it's forced to 1.
    '''
    return [1 if not round(frac) else round(frac) for frac in frac_list]

def l1_norm_diff(GL1, GL2):
    '''
    Computes the L1 norm between group list GL1 and GL2. They must be of the
    same size both in terms of the list and each individual group.
    Assume GL1 is the original group list and GL2 is the reduced one.
    '''
    assert len(GL1) == len(GL2) and GL1 and GL2, 'GL1 and GL2 must be ' \
                                                 'non-empty equal size.'
    for i in range(len(GL1)):
        assert len(GL1[i]) == len(GL2[i]), 'Groups should be equal size.'

    l1_norm = 0
    for i in range(len(GL1)):
        G1, G2 = np.array(GL1[i]), np.array(GL2[i])
        traffic_vol = sum(G1) if len(GL1) > 1 else 1
        l1_norm += traffic_vol * np.linalg.norm((G2 / sum(G2) - G1 / sum(G1)),
                                                ord=1)
    return l1_norm

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
    group, as per equation 2 in the EuroSys WCMP paper. It is used in the group
    reduction algoritm (Algorithm 1 in the paper).

    G: original group before reduction.
    G_prime: final group after reduction.
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
    def __init__(self, groups, table_limit=TABLE_LIMIT):
        '''
        groups: input groups of intended weights reflecting desired traffic
                distribution.

        orig_groups: a list of lists, where each element list is a set of
                     weights for the corresponding group.
        int_groups: original groups after rounding to integers.
        table_limit: upper bound of the group entries on the switch. Note that
                     this does not have to be the physical limit, but can also
                     be the available headroom left.
        '''
        # Strips all the zeroes in each group, they can cause division errors
        # when computing port oversub.
        self._orig_groups = copy.deepcopy([[w for w in g if w > 0] \
                                           for g in groups])
        self._int_groups = list(map(frac2int_round,
                                    copy.deepcopy(self._orig_groups)))
        self._groups = self._int_groups if USE_INT_INPUT_GROUPS else \
                                           self._orig_groups
        self._table_limit = table_limit
        self._final_groups = None
        self._unstripped_groups = copy.deepcopy(groups)
        # Auxiliary map to store the pre-sort location of each int group in the
        # input. This is used for re-constructing the unstripped groups.
        self._presort_group_loc = {}
        # The order of pre-sort `self._int_groups` stays the same as unstripped
        # groups.
        for idx, g_int in enumerate(self._int_groups):
            self._presort_group_loc[tuple(g_int)] = idx

    def get_input_groups(self):
        '''
        Returns the unstripped input groups.
        '''
        return self._unstripped_groups

    def get_output_groups(self):
        '''
        Returns the output groups.
        '''
        return self._final_groups

    def get_table_limit(self):
        '''
        Returns the group table limit.
        '''
        return self._table_limit

    def get_table_util(self, percentage=False, final_groups=None):
        '''
        Returns the group table utilization.

        percentage: If true, returns utilization in percentage, otherwise
                    returns raw entries consumed.
        final_groups: directly provided final groups instead of the one stored
                      as a member variable.
        '''
        final_G = final_groups if final_groups else self._final_groups
        sum_entries = sum([sum(g) for g in final_G])
        return sum_entries / self._table_limit if percentage else sum_entries

    def sanitize_and_cache(self, final_groups):
        '''
        Sanitizes the final groups: (1) singleton groups should just be [1], (2)
        make sure all weights are integers, (3) unstrip final_groups and reorder
        them so that they correspond to the original groups in the same order.
        Also caches the sanitized groups and returns them.

        final_groups: the final integer groups to be sanitized.
        '''
        sanitized_groups = [[]] * len(final_groups)
        for idx, final_group in enumerate(final_groups):
            # From the final_group, find its corresponding int_group, and looks
            # up the location of this group pre-sorting. The location is where
            # the unstripped group is.
            loc = self._presort_group_loc[tuple(self._int_groups[idx])]
            sanitized_group = [0] * len(self._unstripped_groups[loc])
            nz_indices = np.nonzero(np.array(self._unstripped_groups[loc]))[0]
            if nz_indices.size != len(final_group):
                print(f'[ERROR] sanitize_and_cache: final group size of '
                      f'{final_group} mismatches num of non-zero weights '
                      f'{nz_indices.size} in original group '
                      f'{self._unstripped_groups[idx]}')
                return None
            if len(final_group) == 1:
                sanitized_group[nz_indices[0]] = 1
            else:
                for idx, w in enumerate(final_group):
                    sanitized_group[nz_indices[idx]] = int(w)
            sanitized_groups[loc] = sanitized_group
        self._final_groups = sanitized_groups
        return sanitized_groups

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

        min_oversub, index, P = float('inf'), -1, len(group_to_reduce)
        for i in range(P):
            oversub = ((group_under_reduction[i] + 1) * sum(group_to_reduce)) \
                      / ((sum(group_under_reduction) + 1) * group_to_reduce[i])
            if min_oversub > oversub:
                min_oversub = oversub
                index = i

        return index

    def _reduce_wcmp_group(self, group_to_reduce, theta_max=1.2):
        '''
        Single group weight reduction algoritm with an oversub limit.
        This is algorithm 1 in the EuroSys WCMP paper.
        '''
        # First initializes final group to ECMP.
        final_group = np.ones(len(group_to_reduce))
        while calc_group_oversub(group_to_reduce, final_group) > theta_max:
            index = self._choose_port_to_update(group_to_reduce, final_group)
            final_group[index] += 1
            if sum(final_group) >= sum(group_to_reduce):
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
        T, P = self._table_limit, len(group_to_reduce)
        final_group = copy.deepcopy(group_to_reduce)
        while sum(final_group) > T:
            non_reducible_size = 0
            # Counts singleton ports, as they cannot be reduced any further.
            for i in range(P):
                if final_group[i] == 1:
                    non_reducible_size += final_group[i]
            # If the group is already ECMP, just give up.
            if non_reducible_size == P:
                break
            # Directly shrinks original weights by `reduction_ratio` so that
            # the final group can fit into T. Note that the denominator should
            # technically be sum(group_to_reduce) - non_reducible_size since the
            # singleton ports should just be left out of reduction. But the
            # algorithm still reduces (to 0) and then always rounds up to 1.
            reduction_ratio = (T - non_reducible_size) / sum(group_to_reduce)
            for i in range(P):
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

        return self.sanitize_and_cache([final_group_2])

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
        # The order of post-sort `self._int_groups` stays the same as
        # `groups_out`.
        self._int_groups = groups_in
        groups_out = copy.deepcopy(groups_in)
        total_size = sum([sum(g) for g in groups_in])

        while total_size > S:
            for i in range(len(groups_in)):
                groups_out[i] = self._reduce_wcmp_group(groups_in[i],
                                                        enforced_oversub)
                total_size = sum([sum(g) for g in groups_out])
                if total_size <= S:
                    return self.sanitize_and_cache(groups_out)
            # Relaxes oversub limit if we fail to fit all groups with the same
            # oversub.
            enforced_oversub += step_size

        return self.sanitize_and_cache(groups_out)

    def google_ssmg(self):
        '''
        WCMP weight reduction for table fitting a set of WCMP groups H into
        size S. Algorithm 3 + 4 in the EuroSys WCMP paper. This is Google's
        implementation.
        '''
        if len(self._int_groups) <= 0:
            print('[ERROR] %s: unexpected number of input groups %s' %
                  GroupReduction.google_ssmg.__name__,
                  len(self._int_groups))
            return []

        pass

    def solve_sssg(self, formulation='L1NORM2', groups_in=None, C=None):
        '''
        Given the input groups and table limit, solve the single switch single
        group (SSSG) optimization problem.

        group_in (optional): input groups to use instead.
        C (optional): table limit to use instead.
        '''
        input_groups = groups_in if groups_in else self._groups
        table_limit = C if C else self._table_limit
        final_groups = []
        if len(input_groups) != 1:
            print('[ERROR] %s: unexpected number of input groups %s' %
                  GroupReduction.solve_sssg.__name__, len(input_groups))
            return []

        try:
            # Initialize a new model
            m = gp.Model("single_switch_single_group")
            m.setParam("LogToConsole", 1 if VERBOSE >= 2 else 0)
            m.setParam("FeasibilityTol", 1e-7)
            m.setParam("IntFeasTol", 1e-8)
            m.setParam("MIPGap", 1e-4)
            #m.setParam("NodefileStart", 0.5)
            m.setParam("NodefileDir", "/tmp")
            m.setParam("Threads", 0)
            m.setParam("TimeLimit", 120)
            #m.setParam("LogFile", "gurobi.log")

            # Construct model
            # L1NORM1 normalizes actual integer weights over table limit.
            # L1NORM2 normalizes actual integer weights over its own group sum.
            if formulation == "L1NORM1":
                m = self._sssg_l1_norm_1(m, input_groups[0], table_limit)
            elif formulation == "L1NORM2":
                m.setParam("NonConvex", 2)
                m.setParam("MIPFocus", 2)
                m = self._sssg_l1_norm_2(m, input_groups[0], table_limit)
            else:
                print("Formulation not recognized!")
                return []

            # Optimize model
            m.optimize()

            group = []
            sol_w = dict()
            for v in m.getVars():
                if 'w_' in v.VarName:
                    group.append(round(v.X))
                    sol_w[v.VarName] = v.X
            PRINTV(1, f'Obj: {m.ObjVal}')
            PRINTV(1, f'wf: {self._groups[0]}')
            PRINTV(1, str(sol_w))
            # Applies a final GCD reduction just in case.
            final_groups.append(frac2int_lossless(group))

            return self.sanitize_and_cache(final_groups)

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
            return []
        except AttributeError:
            print('Encountered an attribute error')
            return []

    def _sssg_l1_norm_1(self, m, group_in, C):
        '''
        Build an MILP formulation using L1 norm as the objective for the single
        switch single group (SSSG) optimization. In L1 norm, actual weights to
        be solved are normalized against the swtich table limit.

        m: pre-built empty model, needs decision vars and constraints.
        group_in: the (single) input group to be reduced.
        C: table limit allowed to be used.
        '''
        # Create variables: wf[i] is the intended (fractional) weight, w[i] is
        # the actual (integral) weight.
        wf, wf_sum, w, u = group_in, sum(group_in), [], []
        for n in range(len(wf)):
            w.append(m.addVar(vtype=GRB.INTEGER, lb=0, ub=C,
                              name="w_" + str(n+1)))
            u.append(m.addVar(vtype=GRB.CONTINUOUS, name="u_" + str(n+1)))

        # Set objective
        m.setObjective(gp.quicksum(u), GRB.MINIMIZE)

        # Add constraint: sum(w) <= table_limit
        m.addConstr(gp.quicksum(w) <= C, "group_size_ub")
        for i in range(len(w)):
            # Add constraint: u[i] >= abs(w[i] / table_limit - wf[i]).
            # Note: '==' can be relaxed to '>=' because the objective is to
            # minimize sum(u[i]).
            m.addConstr(w[i] / C - wf[i] / wf_sum <= u[i])
            m.addConstr(wf[i] / wf_sum - w[i] / C <= u[i])
            # Add constraint only if inputs are already scaled up to integers:
            # w[i] <= wf[i]
            if USE_INT_INPUT_GROUPS:
                m.addConstr(w[i] <= wf[i], "no_scale_up_" + str(1+i))

        return m

    def _sssg_l1_norm_2(self, m, group_in, C):
        '''
        Build an MIQCP formulation using L1 norm as the objective for the single
        switch single group (SSSG) optimization. In L1 norm, actual weights to
        be solved are normalized against the sum of actual weights.

        m: pre-built empty model, needs decision vars and constraints.
        group_in: the (single) input group to be reduced.
        C: table limit allowed to be used.
        '''
        # Create variables: wf[i] is the intended (fractional) weight, w[i] is
        # the actual (integral) weight.
        wf, wf_sum, w, u = group_in, sum(group_in), [], []
        for n in range(len(wf)):
            w.append(m.addVar(vtype=GRB.INTEGER, lb=0, ub=C,
                              name="w_" + str(n+1)))
            u.append(m.addVar(vtype=GRB.CONTINUOUS, name="u_" + str(n+1)))
        z = m.addVar(vtype=GRB.CONTINUOUS, name="z")

        # Set objective
        m.setObjective(gp.quicksum(u), GRB.MINIMIZE)

        # Add constraint: sum(w) <= table_limit.
        m.addConstr(gp.quicksum(w) <= C, "group_size_ub")
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
            if USE_INT_INPUT_GROUPS:
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
        final_groups = copy.deepcopy(self._groups)

        try:
            # Initialize a new model
            m = gp.Model("single_switch_multi_group")
            m.setParam("LogToConsole", 1 if VERBOSE >= 2 else 0)
            m.setParam("FeasibilityTol", 1e-7)
            m.setParam("IntFeasTol", 1e-8)
            m.setParam("MIPGap", 1e-4)
            #m.setParam("NodefileStart", 0.5)
            m.setParam("NodefileDir", "/tmp")
            m.setParam("Threads", 0)
            m.setParam("TimeLimit", 120)
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
            PRINTV(1, 'Obj: %s' % m.ObjVal)
            PRINTV(1, str(sol_w))
            PRINTV(1, 'wf: %s' % self._groups)
            # Applies a final GCD reduction just in case.
            final_groups = list(map(frac2int_lossless, final_groups))

            return self.sanitize_and_cache(final_groups)

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
        # port i of group m, w[i] is the actual (integral) weight.
        C = self._table_limit
        wf, wf_sum = copy.deepcopy(self._groups), [sum(g) for g in self._groups]
        w, u, l1_norm, z = [], [], [], []
        # Iterates over group m.
        for m in range(len(wf)):
            wm, um = [], []
            # Iterates over port i of group m.
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

        # Set objective: sum(wf_sum[m] * l1_norm[m]).
        model.setObjective(gp.LinExpr(wf_sum, l1_norm), GRB.MINIMIZE)

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
                # Add constraint:
                # u[m][i] == abs(w[m][i] / sum(w[m]) - wf[m][i] / sum(wf[m])).
                model.addConstr(w[m][i] * z[m] - wf[m][i] / wf_sum[m] <= u[m][i])
                model.addConstr(wf[m][i] / wf_sum[m] - w[m][i] * z[m] <= u[m][i])
                # Add constraint only if inputs are already scaled up to
                # integers: w[m][i] <= wf[m][i]
                if USE_INT_INPUT_GROUPS:
                    model.addConstr(w[m][i] <= wf[m][i],
                                    "no_scale_up_{}_{}".format(m+1, i+1))

        return model

    def table_carving_ssmg(self, formulation="L1NORM1", parallel=True):
        '''
        Carve the table limit into multiple smaller limits and solve the SSMG
        problem as individual SSSG.

        parallel: True enables parallel solving, which uses all the CPU cores.
                  False means sequential solving.
        '''
        if len(self._groups) <= 0:
            print('[ERROR] %s: unexpected number of input groups %s' %
                  GroupReduction.table_carving_ssmg.__name__, len(self._groups))
            return []
        final_groups = copy.deepcopy(self._groups)

        # Computes per-group table limit. This is proportional to the group
        # traffic volume/weight sum.
        C, wf = self._table_limit, copy.deepcopy(self._groups)
        wf_sums = [sum(g) for g in wf]
        Cg = [floor(wf_sum / sum(wf_sums) * C) for wf_sum in wf_sums]

        try:
            # Step 1: solve SSSG using table carving limits.
            # Not all groups would fully use up the allocated space, so there
            # will be unused entries at the end of this step.
            parallelism = os.cpu_count() if parallel else 1
            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                futures = {executor.submit(self.solve_sssg, formulation,
                                           [G_in], Cg[i]): i
                           for i, G_in in enumerate(wf)}

                for future in as_completed(futures):
                    i = futures[future]
                    group = future.result()
                    if not group[0]:
                        print('[ERROR] %s: final group %s is invalid - %s ' %
                              GroupReduction.table_carving_ssmg.__name__, i,
                              group[0])
                    # Applies a final GCD reduction just in case.
                    final_groups[i] = frac2int_lossless(group[0])

            PRINTV(1, f'Intermediate metric: {l1_norm_diff(wf, final_groups)}')
            PRINTV(1, f'Intermediate groups: {final_groups}')
            PRINTV(1, 'Intermediate table util: %s / %s, unused %s' %
                   (self.get_table_util(final_groups=final_groups), C,
                   C - self.get_table_util(final_groups=final_groups)))

            # Step 2: iteratively reclaim and redistribute unused entries.
            # Goal is to find the group with worst metric and improve it with
            # unused entries.
            per_group_metric = sorted([
                [i, sum(wf[i]) * l1_norm_diff([wf[i]], [final_groups[i]])]
                for i in range(len(wf))], key=lambda x: x[1], reverse=True)
            while len(per_group_metric) > 0:
                # Group with worst metric is ranked first.
                i, last_metric = per_group_metric[0]
                unused = C - self.get_table_util(final_groups=final_groups)
                PRINTV(1, f'Working on group {i} with metric {last_metric}, '
                       f'unused entries {unused}.')
                # Table is full, nothing to be done.
                if unused <= 0:
                    break
                while True:
                    # Group i might have not used up its allocated space. By
                    # simply allowing an extra `unused` entries will lead to
                    # double counting. Therefore, only unused entries from other
                    # groups can be allocated to group i.
                    Cg[i] += unused - (Cg[i] - sum(final_groups[i]))
                    group = self.solve_sssg(formulation, [wf[i]], Cg[i])[0]
                    # Applies a final GCD reduction just in case.
                    group = frac2int_lossless(group)

                    # Inner loop stop criteria 1: metric stops decreasing. Also
                    # pops head of `per_group_metric` and moves on to next. The
                    # worst group cannot be improved further, but the second
                    # worst might.
                    if sum(wf[i]) * l1_norm_diff([wf[i]], [group]) \
                            >= last_metric:
                        per_group_metric.pop(0)
                        PRINTV(1, f'No metric improvement, skip group {i}.')
                        break

                    # Metric of worst group indeed improves. There might be some
                    # leftover entries that can be used to improve the metric
                    # again, but the group that needs this most could be
                    # different now. So we sort the list again and see if we
                    # should continue on the same group.
                    final_groups[i] = group
                    last_metric = sum(wf[i]) * l1_norm_diff([wf[i]],
                                                            [final_groups[i]])
                    per_group_metric[0][1] = last_metric
                    PRINTV(1, f'Group {i} new metric {last_metric}.')
                    per_group_metric = sorted(per_group_metric,
                                              key=lambda x: x[1],
                                              reverse=True)
                    # Inner loop stop criteria 2: group with worst metric has
                    # changed, start over.
                    if per_group_metric[0][1] != i:
                        PRINTV(1, f'New worst group: {per_group_metric[0][0]}.')
                        break

            return self.sanitize_and_cache(final_groups)

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
            return []
        except AttributeError:
            print('Encountered an attribute error')
            return []

if __name__ == "__main__":
    table_limit = 16*1024
    # groups, # port per group, lower bound, upper bound, fraction precision
    g, p, lb, ub, frac_digits = 8, 16, 100, 100000, 3

    #input_groups = [[1.1, 2.1], [3.1, 4.1]]
    input_groups = input_groups_gen(g, p, lb, ub, frac_digits)
    gr = GroupReduction(input_groups, table_limit)
    for method in ['L1NORM1', 'L1NORM2']:
        start = time.time_ns()
        output_groups = gr.table_carving_ssmg(method)
        end = time.time_ns()
        print(f"===== Method: {method} =====")
        print('Input: %s' % input_groups)
        print('Output: %s' % output_groups)
        print(f'L1 Norm: {l1_norm_diff(input_groups, output_groups)}')
        print('Solving time (msec):', (end - start)/10**6)
        print(f'Table util: {gr.get_table_util()} /'
              f' {gr.get_table_limit()}, headroom: '
              f'{gr.get_table_limit() - gr.get_table_util()}')
        print(f"===== End of method: {method} =====")
