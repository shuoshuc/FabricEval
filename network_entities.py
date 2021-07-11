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

class Path:
    '''
    A path represents an abstract link between 2 aggregation blocks.
    It contains a collection of links from nodes of one aggregation block to
    nodes of another aggregation block.
    name: path name
    src_aggr_block: source aggregation block of the path
    dst_aggr_block: destination aggregation block of the path
    '''
    def __init__(self, name, src_aggr_block=None, src_aggr_block=None):
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
