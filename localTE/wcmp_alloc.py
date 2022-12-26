import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice

import proto.te_solution_pb2 as te_sol
from google.protobuf import text_format

from common.common import PRINTV
from localTE.group_reduction import GroupReduction


def loadTESolution(filepath):
    if not filepath:
        return None
    sol = te_sol.TESolution()
    with open(filepath, 'r', encoding='utf-8') as f:
        text_format.Parse(f.read(), sol)
    return sol

def reduceGroups(node, g_type, limit, groups):
    '''
    A helper function that wraps around the GroupReduction class to invoke
    group reduction on each node.

    node: node name.
    g_type: group type, SRC/TRANSIT.
    limit: ECMP table limit.
    groups: a list of pre-reduction groups.

    Returns a tuple of the same structure as the input, except groups are
    post-reduction.
    '''
    weight_vec = [g for (g, _) in groups]
    # ECMP table is split half and half, each half dedicated to a group type.
    gr = GroupReduction(weight_vec, round(limit / 2))
    reduced_vec = gr.table_fitting_ssmg()
    #reduced_vec = gr.solve_ssmg()
    reduced_groups = []
    for i, vec in enumerate(reduced_vec):
        reduced_groups.append((vec, groups[i][1]))
    return (node, g_type, limit, reduced_groups)

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
        if prefix_intent.type not in [te_sol.PrefixIntent.PrefixType.SRC,
                                      te_sol.PrefixIntent.PrefixType.TRANSIT]:
            print(f'[ERROR] PrefixIntentToGroups: unknown prefix type '
                  f'{prefix_intent.type}!')
            return

        tmp_g = {}
        for ne in prefix_intent.nexthop_entries:
            port = self._topo.getPortByName(ne.nexthop_port)
            node, limit = port.getParent().name, port.getParent().ecmp_limit
            group = tmp_g.setdefault((node, prefix_intent.type, limit),
                                     [0] * len(port.getParent()._member_ports))
            group[port.index - 1] = ne.weight
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

    def chunkGroupsIn(self, size):
        '''
        Chunks `self.groups_in` to multiple sub-dicts, each of size `size`.
        Returns a generator.
        '''
        it = iter(self.groups_in)
        for i in range(0, len(self.groups_in), size):
            yield {k: self.groups_in[k] for k in islice(it, size)}

    def run(self):
        '''
        Launches all WCMP workers to reduce groups.
        '''
        # Collect groups to be reduced from each WCMPWorker and merge them into
        # a unified map.
        for worker in self._worker_map.values():
            self.groups_in.update(worker.populateGroups())

        # Run group reduction for each node in parallel. Limit parallelism up to
        # the number of CPU cores. This avoids queueing too much jobs and
        # filling up the internal job queue/message pipe.
        parallelism = os.cpu_count()
        for group_slice in self.chunkGroupsIn(parallelism):
            t = time.time()
            with ProcessPoolExecutor(max_workers=parallelism) as exe:
                futures = {exe.submit(reduceGroups, node, g_type, limit, groups)
                           for (node, g_type, limit), groups \
                           in group_slice.items()}
            for fut in as_completed(futures):
                node, g_type, limit, reduced_groups = fut.result()
                self.groups_out[(node, g_type, limit)] = reduced_groups
            PRINTV(1, f'[reduceGroup] batch complete in {time.time() - t} sec.')

        # For each prefix type and each node, install all groups.
        for (node, g_type, _), groups in self.groups_out.items():
            self._topo.installGroupsOnNode(node, groups, g_type)
