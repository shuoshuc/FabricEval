import itertools

import proto.traffic_pb2 as traffic
from tmgen import TrafficMatrix
from tmgen.models import modulated_gravity_tm, uniform_tm

NETNAME = 'toy3'

def tmgen(tor_level, num_clusters, num_nodes, model):
    '''
    Generates a traffic demand matrix according to `model`.

    Returns the populated traffic proto.

    tor_level: Boolean. False means AggrBlock-level demand.
    num_clusters: number of clusters in the fabric.
    num_nodes: number of S1 nodes per cluster. Only used when tor_level=True.
    model: the type TM to use, can be flat/gravity.
    '''
    if model == 'flat':
        return genFlat(tor_level, num_clusters, num_nodes)

def genFlat(tor_level, num_clusters, num_nodes):
    '''
    Generates a flat traffic demand matrix. If tor_level=True, each src-dst pair
    sees 152 Mbps demand. If false, each src-dst pair sees 312 Mbps demand.
    '''
    tm_proto = traffic.TrafficDemand()
    tm_proto.type = traffic.TrafficDemand.DemandType.LEVEL_TOR if tor_level \
        else traffic.TrafficDemand.DemandType.LEVEL_AGGR_BLOCK

    for i, j in itertools.product(range(1, num_clusters + 1),
                                  range(1, num_clusters + 1)):
        if tor_level:
            # Populate ToR-level demand matrix.
            for u, v in itertools.product(range(1, num_nodes + 1),
                                          range(1, num_nodes + 1)):
                # A ToR cannot send traffic to itself.
                if i == j and u == v:
                    continue
                demand = tm_proto.demands.add()
                demand.src = f'{NETNAME}-c{i}-ab1-s1i{u}'
                demand.dst = f'{NETNAME}-c{j}-ab1-s1i{v}'
                demand.volume_mbps = round(320000 / (num_nodes - 1 +
                                                     (num_clusters - 1) *
                                                     num_nodes))
        else:
            # Populate AggrBlock-level demand matrix.
            if i == j:
                continue
            demand = tm_proto.demands.add()
            demand.src = f'{NETNAME}-c{i}-ab1'
            demand.dst = f'{NETNAME}-c{j}-ab1'
            demand.volume_mbps = round(20000 / (num_clusters - 1))

    return tm_proto
