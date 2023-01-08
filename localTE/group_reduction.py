import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain
from math import ceil, floor, gcd, isclose, sqrt
from typing import List

import gurobipy as gp
import numpy as np
import proto.te_solution_pb2 as te_sol
from gurobipy import GRB

import common.flags as FLAG
from common.common import PRINTV


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

def l1_norm_diff(FG, IG):
    '''
    Computes the L1 norm between group list FG and IG. They must be of the
    same size both in terms of the list and each individual group.
    Assume FG is the original fractional group list and IG is the reduced one.
    '''
    assert len(FG) == len(IG) and FG and IG, 'FG and IG must be ' \
                                             'non-empty equal size.'
    for i in range(len(FG)):
        assert len(FG[i].weights()) == len(IG[i].integer), 'Groups must be ' \
                                                           'equal size.'

    l1_norm = 0
    for i in range(len(FG)):
        G1, G2 = np.array(FG[i].weights()), np.array(IG[i].integer)
        traffic_vol = sum(G1) if len(FG) > 1 else 1
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
    G_sum = sum(G.integer)
    G_prime_sum = sum(G_prime.integer)
    numerator = np.asarray(G_prime.integer) * G_sum
    demominator = np.asarray(G.integer) * G_prime_sum

    if mode == 'max':
        return np.max(numerator / demominator)
    elif mode == 'avg':
        return np.average(numerator / demominator)
    elif mode == 'p50':
        return np.percentile(numerator / demominator, 50)
    else:
        print('Mode %s not recognized!' % mode)
        return None

@dataclass
class Port:
    '''
    A data structure representing a group member port.
    '''
    # location of the port in the group.
    loc: int
    # Port WCMP weight.
    w: int

@dataclass
class Group:
    '''
    A data structure that stores different forms of a group.
    '''
    gid: int
    # Original unstripped weights with 0.
    unstrip: List[float]
    # Stripped weights without 0.
    strip: List[float] = field(init=False)
    # Integer weights without 0.
    integer: List[int] = field(init=False)

    def __post_init__(self):
        # Strips all the zeroes in each group, they can cause division errors
        # when computing port oversub.
        self.strip = [w for w in self.unstrip if w > 0]
        self.integer = frac2int_round(self.strip)

    def weights(self):
        return self.integer if FLAG.USE_INT_INPUT_GROUPS else self.strip

    def prune(self):
        '''
        Prunes a port in group to reduce its size.
        '''

        # A group cannot be empty, so stop when its already a singleton.
        if len(self.integer) <= 1:
            return

        if FLAG.IMPROVED_HEURISTIC:
            # Smart pruning policy: finds the smallest non-zero member and
            # prunes it.
            np_unstrip = np.array(self.unstrip)
            idx_min = np.argmin(np.ma.masked_where(np_unstrip==0, np_unstrip))
            self.unstrip[idx_min] = 0.0
            self.__post_init__()
        else:
            # Default policy: prunes the first non-zero weight in the group.
            nz_idx = next((i for i, x in enumerate(self.unstrip) if x), None)
            self.unstrip[nz_idx] = 0.0
            self.strip.pop(0)
            self.integer.pop(0)

class GroupReduction:
    '''
    GroupReduction runs various solvers to reduce input groups to acceptable
    integer groups that can be directly implemented on switch hardware, based on
    defined problem formulation and constraints.
    '''
    def __init__(self, groups, g_type, table_limit=FLAG.TABLE_LIMIT):
        '''
        Initializes internal fields. In this class and all helper functions,
        all the groups are represented as instances of the Group dataclass.

        groups: input groups of intended weights reflecting desired traffic
                distribution. This is a list of lists.

        g_type: group type, an enum from te_sol.PrefixIntent.PrefixType.

        table_limit: upper bound of the group entries on the switch. Note that
                     this does not have to be the physical limit, but can also
                     be the available headroom left.
        '''
        # Converts the input weight vectors to a list of Group() objects.
        self.groups = [Group(gid=i, unstrip=g) for i, g in enumerate(groups)]
        self.g_type = g_type
        self._table_limit = table_limit

    def reset(self):
        '''
        Resets the input groups to original state so that other reduction
        algorithms can get a clean start. Resetting involes re-init the integer
        weights and sorting the groups by gid.
        '''
        for g in self.groups:
            g.integer = frac2int_round(g.strip)
        self.groups.sort(key=lambda g: g.gid)

    def get_table_limit(self):
        '''
        Returns the group table limit.
        '''
        return self._table_limit

    def get_table_util(self, final_groups, percentage=False):
        '''
        Returns the group table utilization.

        final_groups: a list of Group dataclass instances.
        percentage (optional): If true, returns utilization in percentage,
                               otherwise returns raw entries consumed.
        '''
        sum_entries = sum([sum(g.integer) for g in final_groups])
        return sum_entries / self._table_limit if percentage else sum_entries

    def sanitize(self, final_groups):
        '''
        Sanitizes the final groups: (1) singleton groups should just be [1], (2)
        make sure all weights are integers, (3) unstrip final_groups and reorder
        them so that they correspond to the original groups in the same order.

        final_groups: a list of Group dataclass instances.

        Returns a list of lists, where the reduced group vectors corresponds to
        the input groups in position and dimension.
        '''
        sanitized_groups = [[]] * len(final_groups)
        for final_group in final_groups:
            # Starts each sanitized group from an all-zero vector of the same
            # dimension as the unstripped version.
            sanitized_group = [0] * len(final_group.unstrip)
            nz_indices = np.nonzero(np.array(final_group.unstrip))[0]
            if nz_indices.size != len(final_group.integer):
                print(f'[ERROR] sanitize: final integer group size of'
                      f' {len(final_group.integer)} mismatches num of non-zero '
                      f'weights {nz_indices.size} in unstripped group '
                      f'{final_group.unstrip}')
                return None
            if len(final_group.integer) == 1:
                sanitized_group[nz_indices[0]] = 1
            else:
                # Replaces 0 at the location where unstripped group has a
                # non-zero weight with the reduced weight.
                for idx, w in enumerate(final_group.integer):
                    sanitized_group[nz_indices[idx]] = int(w)
            # Reorders the sanitized groups using each group's gid.
            sanitized_groups[final_group.gid] = sanitized_group
        return sanitized_groups

    def direct_ecmp(self):
        '''
        Directly reduces `self.groups` to ECMP form. This is a heuristic for
        TRANSIT groups. Since they are always ECMP/singleton, there is no need
        to go through the full reduction logic.
        Note: we assume the ECMP groups will not exceed table limit after
        de-duplication. But this function does not de-dup, so that the output
        can have an 1-to-1 mapping with the input.
        '''
        ecmp_groups = [[]] * len(self.groups)
        for group in self.groups:
            raw_vec = np.array(group.unstrip)
            raw_vec[raw_vec != 0] = 1
            ecmp_groups[group.gid] = raw_vec.astype(np.int32).tolist()
        return ecmp_groups

    def _choose_port_to_update(self, group_to_reduce, group_under_reduction):
        '''
        Helper function for the reduce_wcmp_group algorithm.
        Iteratively goes through all ports of the group under reduction,
        returns the index of member port whose weight should be incremented to
        result in least maximum oversub.
        '''
        if len(group_to_reduce.integer) != len(group_under_reduction.integer):
            print(f'[ERROR] {GroupReduction._choose_port_to_update.__name__}: '
                  f'group dimension mismatch {len(group_to_reduce.integer)} '
                  f'{len(group_under_reduction.integer)}.')
            return -1

        min_oversub, index, P = float('inf'), -1, len(group_to_reduce.integer)
        for i in range(P):
            oversub = ((group_under_reduction.integer[i] + 1)
                       * sum(group_to_reduce.integer)) \
                      / ((sum(group_under_reduction.integer) + 1)
                         * group_to_reduce.integer[i])
            if min_oversub > oversub:
                min_oversub = oversub
                index = i

        return index

    def _reduce_wcmp_group(self, group_to_reduce, theta_max=1.2):
        '''
        Single group weight reduction algoritm with an oversub limit.
        This is algorithm 1 in the EuroSys WCMP paper.

        group_to_reduce: an instance of Group dataclass.
        '''
        # First initializes final group to ECMP.
        final_group = copy.deepcopy(group_to_reduce)
        final_group.integer = np.ones(len(group_to_reduce.integer)).tolist()
        while calc_group_oversub(group_to_reduce, final_group) > theta_max:
            index = self._choose_port_to_update(group_to_reduce, final_group)
            final_group.integer[index] += 1
            if sum(final_group.integer) >= sum(group_to_reduce.integer):
                return group_to_reduce

        return final_group

    def table_fitting_sssg(self):
        '''
        WCMP weight reduction for table fitting a single WCMP group into table
        limit. Assuming that we are fitting the integer version of self.groups.
        '''
        if len(self.groups) != 1:
            print(f'[ERROR] {GroupReduction.table_fitting_sssg.__name__}: '
                  f'unexpected number of input groups {len(self.groups)}.')
            return []

        group_to_reduce = self.groups[0]
        T, P = self._table_limit, len(group_to_reduce.integer)
        final_group = copy.deepcopy(group_to_reduce)
        while sum(final_group.integer) > T:
            non_reducible_size = 0
            # Counts singleton ports, as they cannot be reduced any further.
            for i in range(P):
                if final_group.integer[i] == 1:
                    non_reducible_size += final_group.integer[i]
            # If the group is already ECMP, just give up.
            if non_reducible_size == P:
                break
            # Directly shrinks original weights by `reduction_ratio` so that
            # the final group can fit into T. Note that the denominator should
            # technically be sum(group_to_reduce) - non_reducible_size since the
            # singleton ports should just be left out of reduction. But the
            # algorithm still reduces (to 0) and then always rounds up to 1.
            reduction_ratio = (T - non_reducible_size) / \
                sum(group_to_reduce.integer)
            for i in range(P):
                final_group.integer[i] = np.floor(group_to_reduce.integer[i] * \
                                                  reduction_ratio)
                if final_group.integer[i] == 0:
                    final_group.integer[i] = 1

        # In case the previous round-down over-reduced groups, which leaves some
        # headroom in T, we make full use of the headroom while minimizing the
        # oversub.
        remaining_size = int(T - sum(final_group.integer))
        min_oversub = calc_group_oversub(group_to_reduce, final_group)
        final_group_2 = copy.deepcopy(final_group)
        for _ in range(remaining_size):
            index = self._choose_port_to_update(group_to_reduce, final_group)
            final_group.integer[index] += 1
            curr_oversub = calc_group_oversub(group_to_reduce, final_group)
            if min_oversub > curr_oversub:
                final_group_2.integer = copy.deepcopy(final_group.integer)
                min_oversub = curr_oversub

        return self.sanitize([final_group_2])

    def table_fitting_ssmg(self):
        '''
        WCMP weight reduction for table fitting a set of WCMP groups H into
        size S. Algorithm 4 in the EuroSys WCMP paper.
        This algorithm can be stuck if all groups in their ECMP form still
        exceed table limit. Relaxing enforced_oversub does not help in this
        case. It simply gives up and returns the best effort groups.
        '''
        if len(self.groups) <= 0:
            print(f'[ERROR] {GroupReduction.table_fitting_ssmg.__name__}: '
                  f'unexpected number of input groups {len(self.groups)}.')
            return []

        enforced_oversub = 1.00
        step_size = 0.05
        S = self._table_limit
        # Sort groups in descending order of size.
        self.groups.sort(key=lambda g: sum(g.integer), reverse=True)
        # No need for a deep copy because each element group would be replaced
        # with a copy if it gets reduced.
        groups_out = self.groups.copy()
        total_size = sum([sum(g.integer) for g in groups_out])
        # Total size when all groups are ECMP.
        ecmp_size = sum([len(g.integer) for g in groups_out])

        while total_size > S:
            for i in range(len(self.groups)):
                groups_out[i] = self._reduce_wcmp_group(self.groups[i],
                                                        enforced_oversub)
                total_size = sum([sum(g.integer) for g in groups_out])
                if total_size <= S:
                    return self.sanitize(groups_out)
                # Groups have already become ECMP, impossible to continue
                # shrinking. Either give up or perform pruning.
                if total_size == ecmp_size:
                    # Extra condition for pruning: there exists valid groups to
                    # prune. If all groups are already singletons, pruning is a
                    # no-op, should just give up.
                    if FLAG.EUROSYS_MOD and ecmp_size != len(groups_out):
                        # Prune a port in the largest group. Basically like
                        # starting over, but keep the current enforced_oversub.
                        self.groups[0].prune()
                        self.groups.sort(key=lambda g: sum(g.integer),
                                         reverse=True)
                        groups_out = self.groups.copy()
                        ecmp_size = sum([len(g.integer) for g in groups_out])
                    else:
                        return self.sanitize(groups_out)
            # Relaxes oversub limit if we fail to fit all groups with the same
            # oversub.
            enforced_oversub += step_size

        return self.sanitize(groups_out)

    def _google_sssg(self, group, oversub_limit, max_group_size):
        '''
        WCMP weight reduction for table fitting one group into max group size,
        under a given oversub limit. This is Google's SSSG implementation.

        group: input Group dataclass instance.
        oversub_limit: maximum allowed oversub on the input group. Note that it
                       could be possible that final group cannot meet the limit.
        max_group_size: max group size allowed for the final group.

        Returns a reduced Group dataclass instance.
        '''
        # Constructs a weight vector with the location of each weight. The
        # location is needed for later restoration. Not using Group dataclass
        # to keep it lightweight.
        old_weights = [Port(loc, w) for loc, w in enumerate(group.integer)]
        old_weights.sort(key=lambda port: port.w, reverse=True)
        old_weights_sum = sum(group.integer)
        # Constructs an ECMP group out of `old_weights`.
        new_weights = [Port(port.loc, 1) for port in old_weights]
        # Ports are already sorted by their weights in descending order. With
        # each element equal to 1 in `new_weights`, the max oversub of the ECMP
        # group equals to the oversub on the last element.
        max_oversub = old_weights_sum / (old_weights[-1].w * len(new_weights))

        # If both oversub and group size are within limits, just return ECMP.
        if max_oversub <= oversub_limit and len(new_weights) <= max_group_size:
            new_group = copy.deepcopy(group)
            new_group.integer = [1] * len(new_weights)
            return new_group

        smallest_max_oversub = max_oversub
        new_weights_with_smallest_oversub = copy.deepcopy(new_weights)

        max_weight_0 = old_weights[0].w
        if old_weights_sum > max_group_size:
            max_weight_0 = ceil(old_weights[0].w * max_group_size / \
                                old_weights_sum)
        else:
            smallest_max_oversub = 1.0
            new_weights_with_smallest_oversub = old_weights

        # Sets upper limit on the number of loops to 16. Evaluation shows this
        # yields reasonable accuracy.
        MAX_NUM_ITER = 16
        delta = round(max_weight_0 / (MAX_NUM_ITER - 1)) \
            if max_weight_0 >= MAX_NUM_ITER else 1

        # Increments the weight of member with largest weight by delta in each
        # round.
        while new_weights[0].w <= max_weight_0:
            new_weights_sum = new_weights[0].w
            weight_ratio = new_weights[0].w / old_weights[0].w
            max_weight_ratio = weight_ratio
            for i in range(1, len(new_weights)):
                # Reduces the weight of other members proportionally.
                new_weights[i].w = ceil(weight_ratio * old_weights[i].w)
                max_weight_ratio = max(max_weight_ratio,
                                       new_weights[i].w / old_weights[i].w)
                new_weights_sum += new_weights[i].w
            # No need to increment the weights further. Failed to fit.
            if new_weights_sum > max_group_size:
                break
            max_oversub = max_weight_ratio * old_weights_sum / new_weights_sum
            if max_oversub <= oversub_limit:
                new_group = copy.deepcopy(group)
                new_group.integer = [port.w for port in \
                                     sorted(new_weights, key=lambda p: p.loc)]
                return new_group
            if max_oversub < smallest_max_oversub:
                smallest_max_oversub = max_oversub
                new_weights_with_smallest_oversub = copy.deepcopy(new_weights)
            # Increments first member by delta, heading to next iteration.
            new_weights[0].w += delta

        # If we arrive here, it means we could not meet the oversub limit.
        new_group = copy.deepcopy(group)
        new_group.integer = [port.w for port in \
                             sorted(new_weights_with_smallest_oversub,
                                    key=lambda port: port.loc)]
        return new_group

    def google_ssmg(self):
        '''
        WCMP weight reduction for table fitting a set of groups into the table
        limit. Internally, it calls _google_sssg() to reduce each group
        independently.
        '''
        if len(self.groups) <= 0:
            print(f'[ERROR] {GroupReduction.google_ssmg.__name__}: unexpected '
                  f'number of input groups {len(self.groups)}.')
            return []

        # Heuristic: directly returns ECMP/singleton groups for TRANSIT type.
        # No need to go through the full reduction process.
        if self.g_type == te_sol.PrefixIntent.PrefixType.TRANSIT:
            return self.direct_ecmp()

        enforced_oversub = 1.00
        step_size = 0.05
        S = self._table_limit
        group_vol = np.array([sum(g.unstrip) for g in self.groups])
        max_group_sizes = np.array([FLAG.MAX_GROUP_SIZE] * len(self.groups))
        # Heuristic: max group size for each group is the larger of the
        # pre-defined constant and the proportional carved space using group
        # traffic volume.
        if FLAG.IMPROVED_HEURISTIC:
            max_group_sizes = np.maximum(np.floor(group_vol \
                                                  / group_vol.sum() * S),
                                         max_group_sizes)

        # Sort groups in descending order of size.
        self.groups.sort(key=lambda g: sum(g.integer), reverse=True)
        # No need for a deep copy because each element group would be replaced
        # with a copy if it gets reduced.
        groups_out = self.groups.copy()
        group_sizes = np.array([sum(g.integer) for g in groups_out])
        curr_size = sum(group_sizes)
        # Counts # continuous iterations with no progress in reduction.
        stuck_iters = 0

        # If both total group size and per-group size are within limits, just
        # return. Otherwise, run reduction.
        while sum(group_sizes) > S or \
                len(group_sizes[group_sizes > max_group_sizes]):
            for i in range(len(self.groups)):
                groups_out[i] = self._google_sssg(self.groups[i],
                                                  enforced_oversub,
                                                  max_group_sizes[i])
                group_sizes = np.array([sum(g.integer) for g in groups_out])
                if sum(group_sizes) <= S and \
                        not len(group_sizes[group_sizes > max_group_sizes]):
                    return self.sanitize(groups_out)

            # Relaxes oversub limit if we fail to fit all groups with the same
            # oversub.
            enforced_oversub += step_size
            # Reduction is making progress, go to next iteration.
            if sum(group_sizes) < curr_size:
                curr_size = sum(group_sizes)
                stuck_iters = 0
                continue
            # If after an iteration of reduction, total size did not decrease,
            # it means no progress. If this happens 3 times in a row, we deem
            # the reduction stuck and prune a port from the first group to
            # reduce the total size.
            stuck_iters += 1
            if stuck_iters > 3:
                self.groups[0].prune()
                PRINTV(2, f'group {self.groups[0].gid} is pruned. Now has '
                       f'{len(self.groups[0].integer)} members.')
                # Group sizes could change after pruning, so sort again to
                # keep the largest first.
                self.groups.sort(key=lambda g: sum(g.integer), reverse=True)
                groups_out = self.groups.copy()
                # Resets stuck iteration counts after pruning so that we get a
                # fresh start.
                stuck_iters = 0

        return self.sanitize(groups_out)

    def solve_sssg(self, formulation='L1NORM2', groups_in=None, C=None):
        '''
        Given the input groups and table limit, solve the single switch single
        group (SSSG) optimization problem.
        Returns an unsanitized group.

        groups_in (optional): input groups to use instead.
        C (optional): table limit to use instead.
        '''
        input_groups = groups_in if groups_in else self.groups
        table_limit = C if C else self._table_limit
        final_group = copy.deepcopy(input_groups[0])
        if len(input_groups) != 1:
            print(f'[ERROR] {GroupReduction.solve_sssg.__name__}: unexpected '
                  f'number of input groups {len(input_groups)}.')
            return []

        try:
            # Initialize a new model
            m = gp.Model("single_switch_single_group")
            m.setParam("LogToConsole", 1 if FLAG.VERBOSE >= 2 else 0)
            m.setParam("FeasibilityTol", 1e-7)
            m.setParam("IntFeasTol", 1e-8)
            m.setParam("MIPGap", 1e-4)
            #m.setParam("NodefileStart", 0.5)
            m.setParam("NodefileDir", "/tmp")
            m.setParam("Threads", 0)
            m.setParam("TimeLimit", FLAG.GUROBI_TIMEOUT)
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
            PRINTV(2, f'Obj: {m.ObjVal}')
            PRINTV(2, f'wf: {self.groups[0].weights()}')
            PRINTV(2, str(sol_w))
            # Applies a final GCD reduction just in case.
            final_group.integer = frac2int_lossless(group)

            return final_group

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
        group_in: the (single) input group of Group dataclass to be reduced.
        C: table limit allowed to be used.
        '''
        # Create variables: wf[i] is the intended (fractional) weight, w[i] is
        # the actual (integral) weight.
        wf, wf_sum, w, u = group_in.weights(), sum(group_in.weights()), [], []
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
            if FLAG.USE_INT_INPUT_GROUPS:
                m.addConstr(w[i] <= wf[i], "no_scale_up_" + str(1+i))

        return m

    def _sssg_l1_norm_2(self, m, group_in, C):
        '''
        Build an MIQCP formulation using L1 norm as the objective for the single
        switch single group (SSSG) optimization. In L1 norm, actual weights to
        be solved are normalized against the sum of actual weights.

        m: pre-built empty model, needs decision vars and constraints.
        group_in: the (single) input group of Group dataclass to be reduced.
        C: table limit allowed to be used.
        '''
        # Create variables: wf[i] is the intended (fractional) weight, w[i] is
        # the actual (integral) weight.
        wf, wf_sum, w, u = group_in.weights(), sum(group_in.weights()), [], []
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
            if FLAG.USE_INT_INPUT_GROUPS:
                m.addConstr(w[i] <= wf[i], "no_scale_up_" + str(1+i))

        return m

    def solve_ssmg(self, formulation='L1NORM'):
        '''
        Given the input groups and table limit, solve the single switch multi 
        group (SSMG) optimization problem.
        '''
        if len(self.groups) <= 0:
            print(f'[ERROR] {GroupReduction.solve_ssmg.__name__}: unexpected '
                  f'number of input groups {len(self.groups)}.')
            return []
        final_groups = copy.deepcopy(self.groups)

        try:
            # Initialize a new model
            m = gp.Model("single_switch_multi_group")
            m.setParam("LogToConsole", 1 if FLAG.VERBOSE >= 2 else 0)
            m.setParam("FeasibilityTol", 1e-7)
            m.setParam("IntFeasTol", 1e-8)
            m.setParam("MIPGap", 1e-4)
            #m.setParam("NodefileStart", 0.5)
            m.setParam("NodefileDir", "/tmp")
            m.setParam("Threads", 0)
            m.setParam("TimeLimit", FLAG.GUROBI_TIMEOUT)
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
                    # gi is group index, pi is port index.
                    gi, pi = int(split[1]) - 1, int(split[2]) - 1
                    final_groups[gi].integer[pi] = round(v.X)
                    sol_w[v.VarName] = v.X
            PRINTV(1, 'Obj: %s' % m.ObjVal)
            PRINTV(1, str(sol_w))
            PRINTV(1, 'wf:')
            for g in self.groups:
                PRINTV(1, f'{g.integer}')
            # Applies a final GCD reduction just in case.
            for final_group in final_groups:
                final_group.integer = frac2int_lossless(final_group.integer)

            return self.sanitize(final_groups)

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
        # port i of group m, w[m][i] is the corresponding integer weight.
        C = self._table_limit
        wf = [g.weights() for g in self.groups]
        wf_sum = [sum(g) for g in wf]
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
                if FLAG.USE_INT_INPUT_GROUPS:
                    model.addConstr(w[m][i] <= wf[m][i],
                                    "no_scale_up_{}_{}".format(m+1, i+1))

        return model

    def table_carving_ssmg(self, formulation="L1NORM2"):
        '''
        Carve the table limit into multiple smaller limits and solve the SSMG
        problem as individual SSSG.
        '''
        if len(self.groups) <= 0:
            print(f'[ERROR] {GroupReduction.table_carving_ssmg.__name__}: '
                  f'unexpected number of input groups {len(self.groups)}.')
            return []

        # Heuristic: directly returns ECMP/singleton groups for TRANSIT type.
        # No need to go through the full reduction process.
        if self.g_type == te_sol.PrefixIntent.PrefixType.TRANSIT:
            return self.direct_ecmp()

        final_groups = copy.deepcopy(self.groups)
        # Computes per-group table limit. This is proportional to the group
        # traffic volume/weight sum.
        C, wf = self._table_limit, [g.weights() for g in self.groups]
        wf_sums = [sum(g) for g in wf]
        # Carves up the table proportionally, some group limits may be 0.
        Cg = [floor(wf_sum / sum(wf_sums) * C) for wf_sum in wf_sums]
        if 0 in Cg:
            # We rebalance the group limits to avoid 0 sizes, since these groups
            # will never fit. First iterate over the group limits from smallest
            # to largest, scale up 0 sizes to 1. These entries are borrowed from
            # the largest groups.
            Cg_ext = sorted([[i, cg] for i, cg in enumerate(Cg)],
                            key=lambda x: x[1])
            borrowed_entries = 0
            for i, (_, cg) in enumerate(Cg_ext):
                if not cg:
                    Cg_ext[i][1] = 1
                    borrowed_entries += 1
            # Now iterate over the group limits in the reverse order, subtract
            # borrowed entries from the largest groups. To avoid hurting one
            # group too much, we take entries in a round robin fashion. If the
            # borrowed entries are more than the number of groups, we repeat the
            # round robin until all borrowed entries are clear.
            while borrowed_entries:
                for i, (_, cg) in reversed(list(enumerate(Cg_ext))):
                    if cg > 1:
                        Cg_ext[i][1] -= 1
                        borrowed_entries -= 1
                    if not borrowed_entries:
                        break
            Cg = [cg for _, cg in sorted(Cg_ext, key=lambda x: x[0])]

        try:
            # Step 1: solve SSSG using table carving limits.
            # Not all groups would fully use up the allocated space, so there
            # will be unused entries at the end of this step.
            for i, G_in in enumerate(self.groups):
                G_out = self.solve_sssg(formulation, [G_in], Cg[i])
                final_groups[i] = G_out

            PRINTV(2, f'Interm. metric: '
                   f'{l1_norm_diff(self.groups, final_groups)}')
            PRINTV(2, 'Intermediate table util: %s / %s, unused %s' %
                   (self.get_table_util(final_groups=final_groups), C,
                   C - self.get_table_util(final_groups=final_groups)))

            # Table is fully used, no need to proceed to step 2.
            if self.get_table_util(final_groups=final_groups) >= C:
                return self.sanitize(final_groups)

            # Step 2: iteratively reclaims and redistributes unused entries.
            # Sorts groups from worst metric to best, collects unused entries
            # and allocates to the current group. Retries each group only once.
            # Or stops early when all entries are used up.
            per_group_metric = sorted([[i, wf_sums[i] * \
                                        l1_norm_diff([self.groups[i]],
                                                     [final_groups[i]])]
                for i in range(len(wf))], key=lambda x: x[1], reverse=True)

            for i, metric in per_group_metric:
                unused = C - self.get_table_util(final_groups=final_groups)
                PRINTV(2, f'Working on group {i} with metric {metric}, unused'
                       f' entries {unused}.')
                # No more unused entries, stop and return.
                if unused <= 0:
                    break
                # Group i might have not used up its allocated space. By simply
                # allowing an extra `unused` entries will lead to double
                # counting. Therefore, only unused entries from other groups can
                # be allocated to group i.
                Cg[i] += unused - (Cg[i] - sum(final_groups[i].integer))
                G = self.solve_sssg(formulation, [self.groups[i]], Cg[i])
                # Updates final group if metric improves.
                if wf_sums[i] * l1_norm_diff([self.groups[i]], [G]) < metric:
                    final_groups[i] = G

            return self.sanitize(final_groups)

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
            return []
        except AttributeError:
            print('Encountered an attribute error')
            return []

    def solve(self, algorithm):
        '''
        Invokes corresponding group reduction method based on `algorithm`.

        algorithm: one of eurosys[_mod]/google[_new]/carving/gurobi.
        '''
        if algorithm == 'eurosys':
            return self.table_fitting_ssmg()
        elif algorithm == 'eurosys_mod':
            return self.table_fitting_ssmg()
        elif algorithm == 'google':
            return self.google_ssmg()
        elif algorithm == 'google_new':
            return self.google_ssmg()
        elif algorithm == 'carving':
            return self.table_carving_ssmg()
        elif algorithm == 'gurobi':
            return self.solve_ssmg()
        else:
            print(f'[ERROR] unknown group reduction algorithm {algorithm}.')
            return None
