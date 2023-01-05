import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import islice

import proto.te_solution_pb2 as te_sol
from google.protobuf import text_format

import common.flags as FLAG
from common.common import PRINTV
from localTE.group_reduction import GroupReduction

# Shorthand alias for SRC and TRANSIT type constants.
SRC = te_sol.PrefixIntent.PrefixType.SRC
TRANSIT = te_sol.PrefixIntent.PrefixType.TRANSIT

def loadTESolution(filepath):
    if not filepath:
        return None
    sol = te_sol.TESolution()
    with open(filepath, 'r', encoding='utf-8') as f:
        text_format.Parse(f.read(), sol)
    return sol

def chunk(container, size):
    '''
    Chunks `container` to multiple pieces, each of size `size`.
    Returns a generator.
    '''
    it = iter(container)
    for i in range(0, len(container), size):
        yield {k for k in islice(it, size)}

def deDupSize(weight_vec):
    '''
    Returns the total size of weight_vec after de-duplication.

    weight_vec: a list of lists, representing groups.
    '''
    return sum([sum(v) for v in set([tuple(vec) for vec in weight_vec])])

def reduceGroups(node, limit, src_groups, transit_groups):
    '''
    A helper function that wraps around the GroupReduction class to invoke
    group reduction on each node.

    node: node name.
    limit: ECMP table limit.
    src_groups: a list of pre-reduction src groups. Can be None.
    transit_groups: a list of pre-reduction transit groups. Can be None.

    Returns a tuple of the same structure as the input, except groups are
    post-reduction.
    '''
    reduced_src_groups, reduced_transit_groups = [], []
    # # entries consumed by transit groups.
    used = 0

    # Work on TRANSIT groups.
    if transit_groups:
        transit_vec = [g for (g, _) in transit_groups]
        gr = GroupReduction(transit_vec, TRANSIT, round(limit / 2))
        reduced_transit = gr.solve(FLAG.GR_ALGO)
        used = deDupSize(reduced_transit)
        for i, vec in enumerate(reduced_transit):
            reduced_transit_groups.append((vec, transit_groups[i][1]))

    # Work on SRC groups.
    if src_groups:
        src_vec = [g for (g, _) in src_groups]
        # ECMP table is by default split half and half between SRC and TRANSIT.
        # With improved heuristic, SRC groups can use all the rest space after
        # fitting TRANSIT groups. This is a relaxation because TRANSIT groups
        # are usually ECMP.
        gr = GroupReduction(src_vec, SRC, limit - used \
                            if FLAG.IMPROVED_HEURISTIC else round(limit / 2))
        reduced_src = gr.solve(FLAG.GR_ALGO)
        for i, vec in enumerate(reduced_src):
            reduced_src_groups.append((vec, src_groups[i][1]))

    return (node, limit, reduced_src_groups, reduced_transit_groups)

class WCMPWorker:
    '''
    A WCMP worker is responsible for mapping the TE intents targeting an
    AggrBlock that it manages to programmed flows and groups on switches.
    '''
    def __init__(self, topo_obj, te_intent):
        self._target_block = te_intent.target_block
        self._topo = topo_obj
        self._te_intent = te_intent
        # groups holds pre-reduction groups. Each is keyed by PrefixType
        # because groups of different types cannot merge. Example format:
        # {
        #   (node, prefix_type, ecmp_limit): [([w1, w2, w3], vol), ...],
        #   ...
        # }
        self.groups = {}

    def populateGroups(self):
        '''
        Translates the high level TE intents to pre-reduction groups.

        Returns `self.groups` for caller to run group reduction.
        '''
        for prefix_intent in self._te_intent.prefix_intents:
            self.convertPrefixIntentToGroups(prefix_intent)

        # `groups` may contain duplicates. Dedup and updates `self.groups`.
        self.consolidateGroups()

        return self.groups

    def convertPrefixIntentToGroups(self, prefix_intent):
        '''
        Converts a PrefixIntent to a list of (pre-reduction) groups.
        Specifically, this function splits a PrefixIntent and puts ports on the
        same node together to form one group.
        No return, directly modifies class member variables.
        '''
        if prefix_intent.type not in [SRC, TRANSIT]:
            print(f'[ERROR] PrefixIntentToGroups: unknown prefix type '
                  f'{prefix_intent.type}!')
            return

        tmp_g = {}
        for ne in prefix_intent.nexthop_entries:
            port = self._topo.getPortByName(ne.nexthop_port)
            node, limit = port.getParent().name, port.getParent().ecmp_limit
            group = tmp_g.setdefault((node, prefix_intent.type, limit),
                                     [0] * len(port.getParent()._member_ports))
            group[port.index - 1] = abs(ne.weight)
        for node_type_limit, g in tmp_g.items():
            self.groups.setdefault(node_type_limit, []).append((g, sum(g)))

    def consolidateGroups(self):
        '''
        Consolidates groups in `self.groups`: de-duplicates groups on the
        same node, and accumulates total traffic carried by a shared group.
        '''
        for node_type_limit, groups in self.groups.items():
            g_vol = {}
            # Each `group` is a tuple of (weight vector, volume).
            for (group, vol) in groups:
                g_vol.setdefault(tuple(group), 0)
                g_vol[tuple(group)] += vol
            self.groups[node_type_limit] = [(list(k), v) \
                                            for k, v in g_vol.items()]

class WCMPAllocation:
    '''
    WCMP allocation class that handles the intra-cluster WCMP implementation.
    It translates the TE solution to flows and groups that are programmed on
    each switch.
    '''
    def __init__(self, topo_obj, input_path, input_proto=None):
        # A map from AggrBlock name to the corresponding WCMP worker instance.
        self._worker_map = {}
        self.groups_in = {}
        self.groups_out = {}
        # Stores the topology object in case we need to look up an element.
        self._topo = topo_obj
        # Loads the full network TE solution.
        sol_proto = input_proto if input_proto else loadTESolution(input_path)
        for te_intent in sol_proto.te_intents:
            aggr_block = te_intent.target_block
            if not topo_obj.hasAggrBlock(aggr_block):
                print(f'[ERROR] WCMPAllocation init: Target block {aggr_blok} '
                      f'does not exist in this topology!')
                continue
            # Here we assume each cluster contains exactly one aggregation
            # block. Since a worker is supposed to align with an SDN control
            # domain, it manages all the aggregation blocks in a cluster. In
            # this case, it only manages one aggregation block.
            self._worker_map[aggr_block] = WCMPWorker(topo_obj, te_intent)

        # Use single process if invoking Gurobi. Gurobi is able to use all
        # CPU cores, no need to multi-process, which adds extra overhead.
        FLAG.PARALLELISM = 1 if FLAG.GR_ALGO in ['carving', 'gurobi'] else \
            os.cpu_count()

    def run(self):
        '''
        Launches all WCMP workers to reduce groups.
        '''
        # Collect groups to be reduced from each WCMPWorker and merge them into
        # a unified map.
        for worker in self._worker_map.values():
            self.groups_in.update(worker.populateGroups())

        # Creates a set of (node, limit) so that we can later fetch both the SRC
        # and TRANSIT groups to reduce together.
        node_limit_set = set()
        for node, _, limit in self.groups_in.keys():
            node_limit_set.add((node, limit))
        # Run group reduction for each node in parallel.
        for set_slice in chunk(node_limit_set, FLAG.PARALLELISM):
            t = time.time()
            with ProcessPoolExecutor(max_workers=FLAG.PARALLELISM) as exe:
                futures = {exe.submit(reduceGroups, node, limit,
                                      self.groups_in.get((node, SRC, limit)),
                                      self.groups_in.get((node, TRANSIT, limit)))
                           for node, limit in set_slice}
            for fut in as_completed(futures):
                node, limit, reduced_src, reduced_transit = fut.result()
                if len(reduced_src):
                    self.groups_out[(node, SRC, limit)] = reduced_src
                if len(reduced_transit):
                    self.groups_out[(node, TRANSIT, limit)] = reduced_transit
            PRINTV(1, f'{datetime.now()} [reduceGroups] batch complete in '
                   f'{time.time() - t} sec.')

        # For each prefix type and each node, install all groups.
        for (node, g_type, _), groups in self.groups_out.items():
            self._topo.installGroupsOnNode(node, groups, g_type)
