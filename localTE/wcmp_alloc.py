import itertools
import os
from concurrent.futures import ThreadPoolExecutor

import proto.te_solution_pb2 as te_sol
from google.protobuf import text_format

from localTE.group_reduction import GroupReduction


def loadTESolution(filepath):
    if not filepath:
        return None
    sol = te_sol.TESolution()
    with open(filepath, 'r', encoding='utf-8') as f:
        text_format.Parse(f.read(), sol)
    return sol


class WCMPWorker:
    '''
    A WCMP worker is responsible for mapping the TE intents targeting an
    AggrBlock that it manages to programmed flows and groups on switches.
    '''
    def __init__(self, topo_obj, te_intent):
        self._target_block = te_intent.target_block
        self._topo = topo_obj
        self._te_intent = te_sol.TEIntent()
        # Persists a local copy of the slice.
        self._te_intent.CopyFrom(te_intent)
        # groups_in holds pre-reduction groups. groups_out holds post-reduction
        # groups. Each is keyed by PrefixType because groups of different types
        # cannot merge. The value of these maps is another map from node name to
        # a list of lists, each list denotes a group.
        self.groups_in = {
            te_sol.PrefixIntent.PrefixType.SRC: {},
            te_sol.PrefixIntent.PrefixType.TRANSIT: {}
        }
        self.groups_out = {
            te_sol.PrefixIntent.PrefixType.SRC: {},
            te_sol.PrefixIntent.PrefixType.TRANSIT: {}
        }

    def run(self):
        '''
        Translates the high level TE intents to programmed flows and groups.
        '''
        for prefix_intent in self._te_intent.prefix_intents:
            self.convertPrefixIntentToGroups(prefix_intent)

        # `groups` may contain duplicates. Dedup and updates `self.groups_in`.
        self.consolidateGroups()

        # Run group reduction for src and transit groups separately.
        for p_type, g_map in self.groups_in.items():
            # Run group reduction for all groups on a node in one shot.
            for node, groups in g_map.items():
                weight_vec = [g for (g, _) in groups]
                gr = GroupReduction(weight_vec,
                                    self._topo.getNodeByName(node).ecmp_limit)
                reduced_vec = gr.table_fitting_ssmg()
                #reduced_vec = gr.solve_ssmg()
                reduced_groups = []
                for i, vec in enumerate(reduced_vec):
                    reduced_groups.append((vec, groups[i][1]))
                self.groups_out[p_type][node] = reduced_groups

        # For each prefix type and each node, install all groups.
        for g_type, g_map in self.groups_out.items():
            for node, groups in g_map.items():
                self._topo.installGroupsOnNode(node, groups, g_type)

    def convertPrefixIntentToGroups(self, prefix_intent):
        '''
        Converts a PrefixIntent to a list of (pre-reduction) groups.
        Specifically, this function splits a PrefixIntent and puts ports on the
        same node together to form one group.
        No return, directly modifies class member variables.
        '''
        if prefix_intent.type not in self.groups_in:
            print(f'[ERROR] PrefixIntentToGroups: unknown prefix type '
                  f'{prefix_intent.type}!')
            return

        groups = {}
        for ne in prefix_intent.nexthop_entries:
            port = self._topo.getPortByName(ne.nexthop_port)
            node_name = port.getParent().name
            group = groups.setdefault(node_name,
                                      [0] * len(port.getParent()._member_ports))
            group[port.index - 1] = ne.weight
        for node, g in groups.items():
            self.groups_in[prefix_intent.type].setdefault(node, []).append(
                (g, sum(g)))

    def consolidateGroups(self):
        '''
        Consolidates groups in `self.groups_in`: de-duplicates groups on the
        same node, and accumulates total traffic carried by a shared group.
        '''
        for p_type, g_map in self.groups_in.items():
            for node, groups in g_map.items():
                g_vol = {}
                # Each `group` is a tuple of (weight vector, volume).
                for (group, vol) in groups:
                    g_vol.setdefault(tuple(group), 0)
                    g_vol[tuple(group)] += vol
                self.groups_in[p_type][node] = [(list(k), v) \
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

    def run(self):
        '''
        Launches all WCMP workers in parallel to speed up the computation.
        '''
        worker_runner = lambda worker : worker.run()
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # futures contain execution results if the worker returns a value.
            futures = {executor.submit(worker_runner, worker)
                       for worker in self._worker_map.values()}
