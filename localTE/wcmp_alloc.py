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
    def __init__(self, topo_obj, te_intent, block_demands):
        self._target_block = te_intent.target_block
        self._topo = topo_obj
        self._te_intent = te_intent
        # An instance of the BlockDemands dataclass. Contains demands related to
        # this AggrBlock. Always check before accessing, it could be None if
        # there is no demand for this block.
        self._block_demands = block_demands
        # groups holds pre-reduction groups. Each is keyed by PrefixType
        # because groups of different types cannot merge. Example format:
        # {
        #   (node, prefix_type, ecmp_limit): [([w1, w2, w3], vol), ...],
        #   ...
        # }
        self.groups = {}

    def sendIdealTraffic(self):
        '''
        Sends ideal traffic onto all links belonging to this block (including
        outgoing DCN links, but not incoming ones).
        '''
        # [Step 1] Sends ideal DCN traffic based on the TEIntent. This will load
        # up all outgoing S3-S3 links.
        for prefix_intent in self._te_intent.prefix_intents:
            for ne in prefix_intent.nexthop_entries:
                port = self._topo.getPortByName(ne.nexthop_port)
                port.orig_link._ideal_residual -= ne.weight

        # [Step 2] Sends ideal S2-S3 traffic. This will load up all S2->S3 links
        # and S3->S2 links.
        # For the traffic on S2->S3 links, it includes SRC traffic leaving this
        # block and TRANSIT traffic bounded back at S2. The sum of egress
        # traffic on each S3 should equal all traffic sent from all S2. Each S2
        # sends an equal share so the traffic is evenly divided by the number of
        # links.
        # For the traffic on S3->S2 links, it includes DST traffic destinating
        # this block and TRANSIT traffic to be bounded back by S2. The sum of
        # ingress traffic on each S3 should equal all traffic sent to all S2.
        # Each S2 receives an equal share so the traffic is evenly divided by
        # the number of links.
        cluster = self._topo.getAggrBlockByName(self._target_block).getParent()
        s3_nodes = self._topo.findNodesinClusterByStage(cluster.name, 3)
        for s3_node in s3_nodes:
            up_ports = self._topo.findUpFacingPortsOfNode(s3_node.name)
            ingress, egress = 0, 0
            # Sum the ingress and egress traffic volume on S3 node.
            for port in up_ports:
                link_in, link_out = port.term_link, port.orig_link
                ingress += link_in.link_speed - link_in._ideal_residual
                egress += link_out.link_speed - link_out._ideal_residual
            down_ports = self._topo.findDownFacingPortsOfNode(s3_node.name)
            # Equally spreads: (1) the ingress traffic on this S3 to all
            # S2-bound links. (2) the egress traffic on this S3 to all links
            # from S2.
            for port in down_ports:
                link_to_s2, link_from_s2 = port.orig_link, port.term_link
                link_to_s2._ideal_residual -= ingress / len(down_ports)
                link_from_s2._ideal_residual -= egress / len(down_ports)

        # If there is no ToR-level demands for this block, nothing can be done
        # for the S1-S2 links, call it a day.
        if not self._block_demands:
            PRINTV(2, f'sendIdealTraffic: {self._target_block} has no demand.')
            return

        # [Step 3] Sends ideal S1-S2 traffic and the internal demand portion of
        # S2-S3 traffic. This will load up all S1->S2 and S2->S1 links. It also
        # adds to the existing S2->S3 and S3->S2 link loads.
        # For egress demand traffic, src ToR is located in this block, the dst
        # is elsewhere. So we only need to load up the S1->S2 links because the
        # S2-S3 links are already processed above as part of the block demands.
        # For ingress demand traffic, dst ToR is located in this block, the src
        # is elsewhere. So we only need to load up the S2->S1 links.
        # For internal demands, they are not exposed to globalTE, hence the load
        # is not reflected on S2-S3 links. We process both the S1->S2 and S2->S1
        # links like above. In addition, each S2 node evenly spreads its portion
        # of the internal demand to load up the S2-S3 links.
        for (src_tor, _), vol in self._block_demands.src_only.items():
            up_ports = self._topo.findUpFacingPortsOfNode(src_tor)
            for port in up_ports:
                link_to_s2 = port.orig_link
                link_to_s2._ideal_residual -= vol / len(up_ports)
        for (_, dst_tor), vol in self._block_demands.dst_only.items():
            up_ports = self._topo.findUpFacingPortsOfNode(dst_tor)
            for port in up_ports:
                link_from_s2 = port.term_link
                link_from_s2._ideal_residual -= vol / len(up_ports)
        s2_nodes = self._topo.findNodesinClusterByStage(cluster.name, 2)
        for (src_tor, dst_tor), vol in self._block_demands.src_dst.items():
            src_up_ports = self._topo.findUpFacingPortsOfNode(src_tor)
            dst_up_ports = self._topo.findUpFacingPortsOfNode(dst_tor)
            for port in src_up_ports:
                link_to_s2 = port.orig_link
                link_to_s2._ideal_residual -= vol / len(src_up_ports)
            for port in dst_up_ports:
                link_from_s2 = port.term_link
                link_from_s2._ideal_residual -= vol / len(dst_up_ports)
            # Equal demand sent to each S2.
            vol_per_s2 = vol / len(s2_nodes)
            for s2_node in s2_nodes:
                s2_up_ports = self._topo.findUpFacingPortsOfNode(s2_node.name)
                for port in s2_up_ports:
                    link_to_s3, link_from_s3 = port.orig_link, port.term_link
                    link_to_s3._ideal_residual -= vol_per_s2 / len(s2_up_ports)
                    link_from_s3._ideal_residual -= vol_per_s2 / len(s2_up_ports)

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
    def __init__(self, topo_obj, traffic_obj, input_path, input_proto=None):
        # A map from AggrBlock name to the corresponding WCMP worker instance.
        self._worker_map = {}
        self.groups_in = {}
        self.groups_out = {}
        # Stores the topology object in case we need to look up an element.
        self._topo = topo_obj
        # Stores the traffic object in case we need to look up a demand.
        self._traffic = traffic_obj
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
            self._worker_map[aggr_block] = WCMPWorker(topo_obj, te_intent,
                self._traffic.getBlockDemands(aggr_block))

        # Use single process if invoking Gurobi. Gurobi is able to use all
        # CPU cores, no need to multi-process, which adds extra overhead.
        FLAG.PARALLELISM = 1 if FLAG.GR_ALGO in ['carving', 'gurobi'] else \
            os.cpu_count()

    def run(self):
        '''
        Launches all WCMP workers to reduce groups.
        '''
        # Generates ideal link utilization of the entire fabric.
        for worker in self._worker_map.values():
            worker.sendIdealTraffic()

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
