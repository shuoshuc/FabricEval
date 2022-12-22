import copy
import ipaddress

import numpy as np
import proto.te_solution_pb2 as te_sol
import proto.topology_pb2 as topo
from google.protobuf import text_format


def loadTopo(filepath):
    if not filepath:
        return None
    network = topo.Network()
    with open(filepath, 'r', encoding='utf-8') as f:
        text_format.Parse(f.read(), network)
    return network

def filterPathSetWithSeg(path_set, path_segment):
    '''
    Filters the given `path_set` and drops all the paths that do not contain
    `path_segment`. Returns the reduced path_set.

    path_segment: a tuple of (src, dst) that represents a segment of a path.
    '''
    filtered_path_set = {}
    for path, segs in path_set.items():
        if path_segment in segs:
            filtered_path_set[path] = segs
    return filtered_path_set

class Port:
    '''
    A port represents a physical network port on a switch node, also one end
    of a physical link.
    name: port name
    orig_link: originating link of which this port is a source port.
    term_link: terminating link of which this port is a destination port.
    port_speed: speed the port is running at.
    dcn_facing: whether a port is facing the Data Center Network (DCN).
    index: index of the port.
    '''
    def __init__(self, name, orig_link=None, term_link=None, speed=None,
                 dcn_facing=None, index=None):
        self.name = name
        self.index = index
        self.orig_link = orig_link
        self.term_link = term_link
        self.port_speed = speed
        self.dcn_facing = dcn_facing
        # parent node this port belongs to.
        self._parent_node = None

    def setParent(self, node):
        self._parent_node = node

    def setOrigLink(self, orig_link):
        self.orig_link = orig_link

    def setTermLink(self, term_link):
        self.term_link = term_link

    def getParent(self):
        return self._parent_node


class Link:
    '''
    A link represents a physical network (unidi) link that connects 2 ports.
    name: link name
    src_port: source port of the link.
    dst_port: destination port of the link.
    link_speed: speed the link is running at (in bps). Speed will be
                auto-negotiated.
    dcn_facing: true iff both ends of the link are dcn facing ports.
    '''
    def __init__(self, name, src_port=None, dst_port=None, speed=None,
                 dcn_facing=None):
        self.name = name
        self.src_port = src_port
        self.dst_port = dst_port
        self.link_speed = speed
        self.dcn_facing = dcn_facing
        # Ideal (w/o precision loss) remaining available capacity on the link.
        self._ideal_residual = speed
        # Real (w/ precision loss) remaining available capacity on the link.
        self._real_residual = speed
        self._parent_path = None

    def setParent(self, path):
        self._parent_path = path

    def resetIdealResidual(self):
        self._ideal_residual = 0

    def resetRealResidual(self):
        self._real_residual = 0


class Node:
    '''
    A node represents a physical switch/node located in an aggregation block.
    name: node name
    stage: stage of the node in the network, e.g., stage 1 means ToR.
    index: index of the node among the nodes of the same stage.
    flow_limit: number of flows the flow table can hold.
    ecmp_limit: number of ECMP entries the ECMP table can hold.
    group_limit: number of ECMP entries allowed to be used by each group.
    host_prefix: aggregated IPv4 prefix for hosts under the node (must be ToR).
    mgmt_prefix: aggregated IPv4 prefix for management interfaces on the node
                 (must be ToR).
    '''
    def __init__(self, name, stage=None, index=None, flow_limit=12288,
                 ecmp_limit=16384, group_limit=128, host_prefix=None,
                 mgmt_prefix=None):
        self.name = name
        self.stage = stage
        self.index = index
        self.flow_limit = flow_limit
        self.ecmp_limit = ecmp_limit
        self.group_limit = group_limit
        # Host prefix in ipaddress module format.
        self.host_prefix = host_prefix
        # Management prefix in ipaddress module format.
        self.mgmt_prefix = mgmt_prefix
        # physical member ports on this node.
        self._member_ports = []
        # parent aggregation block this node belongs to.
        self._parent_aggr_block = None
        # parent cluster this node belongs to.
        self._parent_cluster = None
        # All currently installed groups.
        self._groups = {
            te_sol.PrefixIntent.PrefixType.SRC: [],
            te_sol.PrefixIntent.PrefixType.TRANSIT: []
        }
        # Current ECMP table usage, should equal sum of all groups.
        self.ecmp_used = 0

    def setParent(self, aggr_block, cluster):
        self._parent_aggr_block = aggr_block
        self._parent_cluster = cluster

    def addMember(self, port):
        self._member_ports.append(port)

    def getParentAggrBlock(self):
        return self._parent_aggr_block

    def getParentCluster(self):
        return self._parent_cluster

    def getECMPUtil(self):
        return self.ecmp_used / self.ecmp_limit

    def installGroups(self, groups, group_type):
        '''
        Installs groups on node, and updates ECMP table space used. Groups
        completely overwrites the old ones of the same type.
        '''
        self._groups[group_type] = copy.deepcopy(groups)
        self.ecmp_used = sum([sum(g) for G_by_type in self._groups.values() \
                              for (g, _) in G_by_type])

class AggregationBlock:
    '''
    An aggregation block represents a collection of nodes interconnected to
    form a larger switch (aggregation switch).
    name: aggregation block name
    '''
    def __init__(self, name):
        self.name = name
        # physical member nodes on this aggregation block.
        self._member_nodes = []
        # parent cluster this aggregation block belongs to.
        self._parent_cluster = None

    def setParent(self, cluster):
        self._parent_cluster = cluster

    def addMember(self, node):
        self._member_nodes.append(node)


class Path:
    '''
    A path represents an abstract link between 2 aggregation blocks.
    It contains a collection of links from nodes of one aggregation block to
    nodes of another aggregation block.
    name: path name
    src_aggr_block: source aggregation block of the path
    dst_aggr_block: destination aggregation block of the path
    capacity: capacity of path, equals sum of all member link capacity.
    '''
    def __init__(self, name, src_aggr_block=None, dst_aggr_block=None,
                 capacity=None):
        self.name = name
        self.src_aggr_block = src_aggr_block
        self.dst_aggr_block = dst_aggr_block
        self.capacity = capacity
        self.available_capacity = 0
        # physical member links contained in this path.
        self._member_links = []

    def addMember(self, link):
        self._member_links.append(link)


class Cluster:
    '''
    A cluster represents a collection of aggregation blocks managed by the
    same SDN control domain. It often maps to a pool of servers that is the
    building block of data center fabrics.
    name: cluster name
    '''
    def __init__(self, name):
        self.name = name
        # member aggregation blocks contained in this cluster.
        self._member_aggr_blocks = []
        # member ToRs contained in this cluster.
        self._member_tors = []

    def addMember(self, aggr_block):
        self._member_aggr_blocks.append(aggr_block)

    def addMemberToR(self, tor):
        self._member_tors.append(tor)


class Topology:
    '''
    Topology class that represents a network. It contains physical network
    entities and abstract entities.
    '''
    def __init__(self, input_path, input_proto=None):
        '''
        input_path: path to the topology protobuf. Must provide.
        input_proto: raw protobuf of the topology. Optional. If set, ignores
                     `input_path`.
        '''
        self._clusters = {}
        self._aggr_blocks = {}
        self._nodes = {}
        self._ports = {}
        self._paths = {}
        self._links = {}
        # A map from (src_ag_block, dst_ag_block) pair to link.
        ag_block_link_map = {}
        # parse input topology and populate this topology.
        proto_net = input_proto if input_proto else loadTopo(input_path)
        for cluster in proto_net.clusters:
            cluster_obj = Cluster(cluster.name)
            self._clusters[cluster.name] = cluster_obj
            for ag_block in cluster.aggr_blocks:
                ag_block_obj = AggregationBlock(ag_block.name)
                self._aggr_blocks[ag_block.name] = ag_block_obj
                cluster_obj.addMember(ag_block_obj)
                ag_block_obj.setParent(cluster_obj)
                for node in ag_block.nodes:
                    node_obj = Node(node.name, node.stage, node.index,
                                    node.flow_limit, node.ecmp_limit,
                                    node.group_limit)
                    self._nodes[node.name] = node_obj
                    ag_block_obj.addMember(node_obj)
                    node_obj.setParent(ag_block_obj, cluster_obj)
                    for port in node.ports:
                        port_obj = Port(port.name, speed=port.port_speed_mbps,
                                        dcn_facing=port.dcn_facing,
                                        index=port.index)
                        self._ports[port.name] = port_obj
                        node_obj.addMember(port_obj)
                        port_obj.setParent(node_obj)
            for tor in cluster.nodes:
                if tor.stage != 1:
                    print('[ERROR] Topology parsing: ToR {} cannot be stage '
                          '{}!'.format(tor.name, tor.stage))
                    return
                host_prefix, mgmt_prefix = None, None
                if (tor.host_prefix and tor.host_mask != 0 and
                    tor.host_mask < 32):
                    host_prefix = ipaddress.ip_network(tor.host_prefix + '/' +
                                                       str(tor.host_mask))
                if (tor.mgmt_prefix and tor.mgmt_mask != 0 and
                    tor.mgmt_mask < 32):
                    mgmt_prefix = ipaddress.ip_network(tor.mgmt_prefix + '/' +
                                                       str(tor.mgmt_mask))
                tor_obj = Node(tor.name, tor.stage, tor.index, tor.flow_limit,
                               tor.ecmp_limit, tor.group_limit, host_prefix,
                               mgmt_prefix)
                self._nodes[tor.name] = tor_obj
                tor_obj.setParent(None, cluster_obj)
                cluster_obj.addMemberToR(tor_obj)
                for port in tor.ports:
                    port_obj = Port(port.name, speed=port.port_speed_mbps,
                                    dcn_facing=port.dcn_facing,
                                    index=port.index)
                    self._ports[port.name] = port_obj
                    tor_obj.addMember(port_obj)
                    port_obj.setParent(tor_obj)
        for link in proto_net.links:
            if (link.src_port_id not in self._ports or
                link.dst_port_id not in self._ports):
                print('[ERROR] Topology parsing: link {} has at least one '
                      'port not found! src: {}, dst: {}'.format(link.name,
                          link.src_port_id, link.dst_port_id))
                return
            src_port_obj = self._ports[link.src_port_id]
            dst_port_obj = self._ports[link.dst_port_id]
            link_obj = Link(link.name, src_port_obj, dst_port_obj,
                            min(link.link_speed_mbps, src_port_obj.port_speed,
                                dst_port_obj.port_speed),
                            src_port_obj.dcn_facing and dst_port_obj.dcn_facing)
            self._links[link.name] = link_obj
            src_port_obj.setOrigLink(link_obj)
            dst_port_obj.setTermLink(link_obj)
            src_ag_block = self.findAggrBlockOfPort(link.src_port_id)
            dst_ag_block = self.findAggrBlockOfPort(link.dst_port_id)
            # Non-DCN links cannot belong to a path.
            if link_obj.dcn_facing and src_ag_block and dst_ag_block:
                ag_block_link_map.setdefault((src_ag_block, dst_ag_block),
                                             []).append(link_obj)
        for path in proto_net.paths:
            if (path.src_aggr_block not in self._aggr_blocks or
                path.dst_aggr_block not in self._aggr_blocks):
                print('[ERROR] Topology parsing: path {} has at least one '
                      'aggr_block not found! src: {}, dst: {}'.format(path.name,
                          path.src_aggr_block, path.dst_aggr_block))
                return
            src_ag_block_obj = self._aggr_blocks[path.src_aggr_block]
            dst_ag_block_obj = self._aggr_blocks[path.dst_aggr_block]
            path_obj = Path(path.name, src_ag_block_obj, dst_ag_block_obj,
                            path.capacity_mbps)
            self._paths[path.name] = path_obj
            # If src-dst is not found in the link map, there is no DCN link
            # between the pair, so just skip.
            if (src_ag_block_obj, dst_ag_block_obj) not in ag_block_link_map:
                continue
            for l in ag_block_link_map[(src_ag_block_obj, dst_ag_block_obj)]:
                path_obj.addMember(l)
                l.setParent(path_obj)
                path_obj.available_capacity += l.link_speed
            if path_obj.capacity != path_obj.available_capacity:
                print('[ERROR] Topology parsing: path {} has capacity {} and '
                      'available_capacity {}.'.format(path.name,
                          path_obj.capacity, path_obj.available_capacity))
                return

    def numClusters(self):
        '''
        Returns number of clusters in this topology.
        '''
        return len(self._clusters)

    def numNodes(self):
        '''
        Returns number of nodes in this topology.
        '''
        return len(self._nodes)

    def numPorts(self):
        '''
        Returns number of ports in this topology.
        '''
        return len(self._ports)

    def numLinks(self):
        '''
        Returns number of links in this topology.
        '''
        return len(self._links)

    def getAllPaths(self):
        '''
        Returns all the paths.
        '''
        return self._paths

    def getAllAggrBlocks(self):
        '''
        Returns all the aggregation blocks.
        '''
        return self._aggr_blocks

    def getPortByName(self, port_name):
        '''
        Looks up the port object of the given port name.
        '''
        if port_name not in self._ports:
            print('[ERROR] {}: Input port {} does not exist in this topology!'
                  .format('getPortByName', port_name))
            return None
        return self._ports[port_name]

    def getNodeByName(self, node_name):
        '''
        Looks up the node object of the given node name.
        '''
        if node_name not in self._nodes:
            print('[ERROR] {}: Input node {} does not exist in this topology!'
                  .format('getNodeByName', node_name))
            return None
        return self._nodes[node_name]

    def findPeerPortOfPort(self, port_name):
        '''
        Looks up the peer port of the given port, returns the port object.
        '''
        if port_name not in self._ports:
            print('[ERROR] {}: Input port {} does not exist in this topology!'
                  .format('Find peer port', port_name))
            return None
        port_obj = self._ports[port_name]
        if not port_obj.orig_link or not port_obj.term_link:
            print('[ERROR] {}: Input port {} is missing orig_link or term_link.'
                  .format('Find peer port', port_name))
            return None
        assert port_obj.orig_link.dst_port == port_obj.term_link.src_port
        return port_obj.orig_link.dst_port

    def findAggrBlockOfPort(self, port_name):
        '''
        Looks up the parent aggregation block of the given port. Returns the
        aggr_block topology.
        '''
        if port_name not in self._ports:
            print('[ERROR] {}: Input port {} does not exist in this topology!'
                  .format('findAggrBlockOfPort', port_name))
            return None
        port_obj = self._ports[port_name]
        return port_obj.getParent().getParentAggrBlock()

    def findCapacityOfPath(self, path_name):
        '''
        Looks up the available capacity of the given path, returns an integer
        in Mbps. Note that paths are uni-directional.
        '''
        if path_name not in self._paths:
            print('[ERROR] {}: Path {} does not exist in this topology!'
                  .format('Find path capacity', path_name))
            return -1
        return self._paths[path_name].available_capacity

    def findCapacityOfPathTuple(self, path_tuple):
        '''
        Looks up the available capacity of the given path, returns an integer
        in Mbps. This version works with 2-segment transit paths as well as
        direct path. len(path_tuple) == 2 means direct path.
        '''
        capacity = float("inf")
        path_list = []
        if len(path_tuple) == 2:
            path_list = [f'{path_tuple[0]}:{path_tuple[1]}']
        elif len(path_tuple) == 3:
            path_list = [f'{path_tuple[0]}:{path_tuple[1]}',
                         f'{path_tuple[1]}:{path_tuple[2]}']
        else:
            print(f'[ERROR] findCapacityOfPath: Path {path_tuple} is illegal!')
            return -1

        for path_name in path_list:
            capacity = min(capacity, self.findCapacityOfPath(path_name))
        return capacity

    def hasAggrBlock(self, aggr_block_name):
        '''
        Returns true if the given aggr_block can be found in the topology.
        '''
        return (aggr_block_name in self._aggr_blocks)

    def hasPath(self, src, dst):
        '''
        Returns true if there exists a path that corresponds to the given
        src-dst pair.
        '''
        return (f"{src}:{dst}" in self._paths)

    def findHostPrefixOfToR(self, tor_name):
        '''
        Returns the IP prefix of given ToR.
        '''
        if tor_name not in self._nodes:
            print('[ERROR] {}: Node {} does not exist in this topology!'
                  .format('Find host prefix', tor_name))
            return None
        tor = self._nodes[tor_name]
        if tor.stage != 1:
            print('[ERROR] {}: Node {} stage={} is not a ToR!'
                  .format('Find host prefix', tor_name, tor.stage))
            return None
        return tor.host_prefix

    def findAggrBlockOfToR(self, tor_name):
        '''
        Looks up the parent AggrBlock of the given ToR.
        Note: Strictly speaking, a ToR's parent is a cluster, not AggrBlock. But
              for TE purposes, we just want to find the AggrBlock of that parent
              cluster.
        '''
        if tor_name not in self._nodes:
            print('[ERROR] {}: Node {} does not exist in this topology!'
                  .format('findAggrBlockOfToR', tor_name))
            return None
        tor = self._nodes[tor_name]
        if tor.stage != 1:
            print('[ERROR] {}: Node {} stage={} is not a ToR!'
                  .format('findAggrBlockOfToR', tor_name, tor.stage))
            return None
        return tor.getParentCluster()._member_aggr_blocks[0]

    def findOrigPathsOfAggrBlock(self, src):
        '''
        Returns a dict of {path_name: (src, dst)} such that all these paths
        originate from aggregation block src.

        src: name of aggregation block.
        '''
        path_dict = {}
        for path_name, path_obj in self._paths.items():
            if src == path_obj.src_aggr_block.name:
                path_dict[path_name] = (src, path_obj.dst_aggr_block.name)
        return path_dict

    def findTermPathsOfAggrBlock(self, dst):
        '''
        Returns a dict of {path_name: (src, dst)} such that all these paths
        terminate at aggregation block dst.

        dst: name of aggregation block.
        '''
        path_dict = {}
        for path_name, path_obj in self._paths.items():
            if dst == path_obj.dst_aggr_block.name:
                path_dict[path_name] = (path_obj.src_aggr_block.name, dst)
        return path_dict

    def findPathSetOfAggrBlockPair(self, src, dst):
        '''
        Returns a path set for src-dst pair.
        A path set is a dict of {end2end path: [path segments]}. For example,
        {
          (s, t): [(s, t)],
          (s, m, t): [(s, m), (m, t)],
          ...
        }
        All s, m, t are names of aggregation blocks.
        NB: we do not assume the path to be longer than 2-hop.
        '''
        path_set = {}
        # A direct path exists, add to path set.
        if self.hasPath(src, dst):
            path_set[(src, dst)] = [(src, dst)]
        orig_paths = self.findOrigPathsOfAggrBlock(src)
        term_paths = self.findTermPathsOfAggrBlock(dst)
        # If the terminating node of an orig_path overlaps with the orignating
        # node of a term_path, this is a 2-hop path from src to dst. The direct
        # path is automatically ignored since there is no (dst, dst) path. But
        # it has been handled aboved.
        for _, m in orig_paths.values():
            if f"{m}:{dst}" in term_paths:
                path_set[(src, m, dst)] = [(src, m), (m, dst)]
        return path_set

    def findLinksOfPath(self, path_name):
        '''
        Returns a list of link objects in path.
        '''
        if path_name not in self._paths:
            print(f'[ERROR] findLinksOfPath(): path {path_name} not found.')
            return None
        return self._paths[path_name]._member_links

    def distributeFlows(self, path_map):
        '''
        Distributes the MCF solution for a commodity into flows on each physical
        link, links will accumulate the flows placed on them.

        Also returns a distribution of flows of format:
        {
            (s, t): {port1: w1, port2: w2},
            (s, m): {port3: w3, port4: w4},
            (m, t): {port5: w5, port6: w6}
        }
        s, m, t are AggregationBlock names, port[1-6] are physical port names.

        path_map: a map from multi-segment path name to traffic volume (in Mbps)
                  carried on it.

        Note: we do not expect paths longer than 2 hops.
        '''
        flow_dist = {}
        for path_name, traffic_vol in path_map.items():
            node_list = path_name.split(":")
            if len(node_list) != 2 and len(node_list) != 3:
                print(f'[ERROR] distributeFlows(): path {path_name} has '
                      '{len(node_list) - 1} hops. Only 1 or 2 hops are legal.')
                return None
            # Enumerates all the single-segment paths. This path is at least a
            # direct path.
            single_seg_paths = [(node_list[0], node_list[1])]
            # This path is a 2-hop path.
            if len(node_list) == 3:
                single_seg_paths.append((node_list[1], node_list[2]))
            # Retrives all links on each single-segment path.
            for seg in single_seg_paths:
                links = self.findLinksOfPath(f'{seg[0]}:{seg[1]}')
                # Traffic volume is split via ECMP among links of the same path.
                # This is assuming all links have the same speed.
                t_frac = traffic_vol / len(links)
                for link in links:
                    # Accumulates flow on each link.
                    link._ideal_residual -= t_frac
                    flow_dist.setdefault(seg, {})[link.src_port.name] = t_frac
        return flow_dist

    def dumpIdealLinkUtil(self):
        '''
        Returns a map of all link utilizations normalized to 1. Returned dict is
        sorted from highest link util to lowest.
        NB: returned link util is ideal utilization without precision loss.
        '''
        link_util_map = {}
        for link in self._links.values():
            link_util_map[link.name] = \
                (link.link_speed - link._ideal_residual) / link.link_speed
        return dict(sorted(link_util_map.items(), key=lambda x: x[1],
                           reverse=True))

    def installGroupsOnNode(self, node_name, groups, group_type):
        '''
        Installs groups on node. Groups will be persisted in the node instance
        until next installation. ECMP util will be updated as well. In addition,
        each link will be updated with the traffic load according to the group
        weights.

        node_name: name of the node the groups belong to.
        groups: a list of (group, volume) tuples.
        group_type: SRC or TRANSIT, groups of different types cannot be shared.
        '''
        node = self.getNodeByName(node_name)
        node.installGroups(groups, group_type)
        for (group, vol) in groups:
            # Finds all the ports that carry traffic in the group.
            nz_indices = np.nonzero(np.array(group))[0]
            for i in nz_indices:
                port_name = f'{node_name}-p{i + 1}'
                port = self.getPortByName(port_name)
                # Places traffic fraction of the group on the outgoing link of
                # port (since traffic is outgoing). `_real_residual` can be
                # negative due to oversubscription.
                port.orig_link._real_residual -= group[i] / sum(group) * vol

    def dumpRealLinkUtil(self):
        '''
        Returns a map of all link utilizations normalized to 1. Returned dict is
        sorted from highest link util to lowest.
        NB: returned link util is real utilization with precision loss.
        '''
        link_util_map = {}
        for link in self._links.values():
            link_util_map[link.name] = \
                (link.link_speed - link._real_residual) / link.link_speed
        return dict(sorted(link_util_map.items(), key=lambda x: x[1],
                           reverse=True))
