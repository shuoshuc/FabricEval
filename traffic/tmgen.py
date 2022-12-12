import itertools

import proto.traffic_pb2 as traffic
from tmgen import TrafficMatrix
from tmgen.models import exact_tm, modulated_gravity_tm, uniform_tm

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
        # Generates a flat traffic demand matrix. If tor_level=True, each
        # src-dst pair sees 152 Mbps demand. If false, each src-dst pair sees
        # 80000 Mbps demand.
        if tor_level:
            tm = exact_tm(num_clusters * num_nodes,
                          round(40000 * 8 / (num_nodes * num_clusters - 1)),
                          num_epochs=1)
        else:
            tm = exact_tm(num_clusters, round(20000 * 256 / (num_clusters - 1)),
                          num_epochs=1)
        return genProto(tor_level, num_clusters, num_nodes, tm.at_time(0))
    elif model == 'uniform':
        # Generates a uniform random traffic demand matrix. Each src-dst pair
        # will not exceed the value of a same entry in the flat TM.
        if tor_level:
            tm = uniform_tm(num_clusters * num_nodes, 0,
                            round(40000 * 8 / (num_nodes * num_clusters - 1)),
                            num_epochs=1)
        else:
            tm = uniform_tm(num_clusters, 0,
                            round(40000 * 256 / (num_clusters - 1)),
                            num_epochs=1)
        return genProto(tor_level, num_clusters, num_nodes, tm.at_time(0))

def genProto(tor_level, num_clusters, num_nodes, TM):
    '''
    Returns a traffic proto using the given traffic matrix `TM`.
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
                demand.volume_mbps = round(TM[(i - 1) * 32 + u - 1,
                                              (j - 1) * 32 + v - 1])
        else:
            # Populate AggrBlock-level demand matrix.
            if i == j:
                continue
            demand = tm_proto.demands.add()
            demand.src = f'{NETNAME}-c{i}-ab1'
            demand.dst = f'{NETNAME}-c{j}-ab1'
            demand.volume_mbps = round(TM[i-1, j-1])

    return tm_proto
