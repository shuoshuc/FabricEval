import proto.topology_pb2 as topo
from google.protobuf import text_format

def loadTopo(filepath):
    if not filepath:
        return None
    with open(filepath, 'r') as f:
        network = topo.Network()
        text_format.Parse(f.read(), network)
    return network


class Port:
    '''
    A port represents a physical network port on a switch node, also one end
    of a physical link.
    name: port name
    orig_link: originating link of which this port is a source port.
    term_link: terminating link of which this port is a destination port.
    port_speed: speed the port is running at.
    dcn_facing: whether a port is facing the Data Center Network (DCN).
    '''
    def __init__(self, name, orig_link=None, term_link=None, speed=None,
                 dcn_facing=None):
        self.name = name
        self.orig_link = orig_link
        self.term_link = term_link
        self.port_speed = speed
        self.dcn_facing = dcn_facing
        # parent node this port belongs to.
        self._parent_node = None

    def setParent(node):
        self._parent_node = node

    def setOrigLink(orig_link):
        self.orig_link = orig_link

    def setTermLink(term_link):
        self.term_link = term_link


class Link:
    '''
    A link represents a physical network (unidi) link that connects 2 ports.
    name: link name
    src_port: source port of the link.
    dst_port: destination port of the link.
    link_speed: speed the link is running at (in bps). Speed will be
                auto-negotiated.
    '''
    def __init__(self, name, src_port=None, dst_port=None, speed=None):
        self.name = name
        self.src_port = src_port
        self.dst_port = dst_port
        self.link_speed = speed
        # remaining available capacity on the link.
        self._residual_capacity = speed


class Node:
    '''
    A node represents a physical switch/node located in an aggregation block.
    name: node name
    stage: stage of the node in the network, e.g., stage 1 means ToR.
    index: index of the node among the nodes of the same stage.
    flow_limit: number of flows the flow table can hold.
    ecmp_limit: number of ECMP entries the ECMP table can hold.
    group_limit: number of ECMP entries allowed to be used by each group.
    '''
    def __init__(self, name, stage=None, index=None, flow_limit=12288,
                 ecmp_limit=16384, group_limit=128):
        self.name = name
        self.stage = stage
        self.index = index
        self.flow_limit = flow_limit
        self.ecmp_limit = ecmp_limit
        self.group_limit = group_limit
        # physical member ports on this node.
        self._member_ports = []
        # parent aggregation block this node belongs to.
        self._parent_aggr_block = None

    def setParent(aggr_block):
        self._parent_aggr_block = aggr_block

    def addMember(port):
        self._member_ports.append(port)


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

    def setParent(cluster):
        self._parent_cluster = cluster

    def addMember(node):
        self._member_nodes.append(node)


class Path:
    '''
    A path represents an abstract link between 2 aggregation blocks.
    It contains a collection of links from nodes of one aggregation block to
    nodes of another aggregation block.
    name: path name
    src_aggr_block: source aggregation block of the path
    dst_aggr_block: destination aggregation block of the path
    '''
    def __init__(self, name, src_aggr_block=None, dst_aggr_block=None):
        self.name = name
        self.src_aggr_block = src_aggr_block
        self.dst_aggr_block = dst_aggr_block
        # physical member links contained in this path.
        self._member_links = []
        # remaining available capacity on the link.
        self._available_capacity = 0


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

    def addMember(self, aggr_block):
        self._member_aggr_blocks.append(aggr_block)


class Topology:
    '''
    Topology class that represents a network. It contains physical network
    entities and abstract entities.
    '''
    def __init__(self, input_proto):
        self._clusters = {}
        self._aggr_blocks = {}
        self._nodes = {}
        self._ports = {}
        self._paths = {}
        self._links = {}
        # parse input topology and populate this topology.
        proto_net = loadTopo(input_proto)
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
                    node_obj.setParent(ag_block_obj)
                    for port in node.ports:
                        port_obj = Port(port.name, speed=port.port_speed,
                                        dcn_facing=port.dcn_facing)
                        self._ports[port.name] = port_obj
                        node_obj.addMember(port_obj)
                        port_obj.setParent(node.obj)
        for link in proto_net.links:
            if (link.src_port_id not in self._ports or
                link.dst_port_id not in self._ports):
                print('[ERROR] link {} has at least one port not found! '
                      'src: {}, dst: {}'.format(link.name, link.src_port_id,
                                                link.dst_port_id))
                return
            src_port_obj = self._ports[link.src_port_id]
            dst_port_obj = self._ports[link.dst_port_id]
            link_obj = Link(link.name, src_port_obj, dst_port_obj,
                            link.link_speed)
            self._links[link.name] = link_obj
            src_port_obj.setOrigLink(link_obj)
            dst_port_obj.setTermLink(link_obj)
        for path in proto_net.paths:
            # TODO: path is incomplete
            self._paths[path.name] = Path(path.name)
