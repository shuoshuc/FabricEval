import itertools
from math import floor

import numpy as np
import proto.traffic_pb2 as traffic
from numpy.random import default_rng
from scipy.stats import truncexpon

NETNAME = 'toy3'
# True means the block total ingress should equal its total egress.
EQUAL_INGRESS_EGRESS = True

def tmgen(tor_level, num_clusters, num_nodes, model):
    '''
    Generates a traffic demand matrix according to `model`.

    Returns the populated traffic proto.

    tor_level: Boolean. False means AggrBlock-level demand.
    num_clusters: number of clusters in the fabric.
    num_nodes: number of S1 nodes per cluster. Only used when tor_level=True.
    model: the type TM to use, can be flat/gravity.
    '''
    rng = default_rng()
    size = num_clusters * num_nodes if tor_level else num_clusters
    tm = np.zeros(shape=(size, size))
    if model == 'flat':
        # Generates a flat traffic demand matrix. If tor_level=True, each
        # src-dst pair sees 153 Mbps demand. If false, each src-dst pair sees
        # 80000 Mbps demand.
        tm[tm >= 0] = 40000 * 8 / (num_nodes * num_clusters - 1) if tor_level \
            else 20000 * 256 / (num_clusters - 1)
    elif model == 'uniform':
        # Generates a uniform random traffic demand matrix. Each src-dst pair
        # will not exceed the value of a same entry in the flat TM.
        upper_bound = 40000 * 8 / (num_nodes * num_clusters - 1) if tor_level \
            else 40000 * 256 / (num_clusters - 1)
        for r, c in np.ndindex(tm.shape):
            tm[r, c] = rng.uniform(low=0, high=upper_bound)
    elif model == 'gravity':
        # Generates a traffic demand matrix following the gravity model. Each
        # src-dst pair has a demand proportional to the product of their egress
        # and ingress demands. The block total ingress/egress volume is sampled
        # from a uniform random/exponential/Pareto distribution.
        upper_bound = (40000 * 8 if tor_level else 40000 * 256) * 0.7
        scale = upper_bound / 2
        dist = truncexpon(b=upper_bound/scale, loc=0, scale=scale)
        egress = dist.rvs(size)
        # Set block total ingress to be the same as egress if flag is true.
        # Otherwise, sample another set of values from the same distribution.
        if EQUAL_INGRESS_EGRESS:
            ingress = egress
        else:
            ingress = dist.rvs(size)
            # Rescale the ingress vector so that its sum equals egress. (Total
            # egress and ingress must match).
            ingress *= egress.sum() / ingress.sum()
        # `r` is row vector for src, `c` is column vector for dst.
        for r, c in np.ndindex(tm.shape):
            # Self-loop demand is not allowed.
            if r == c:
                continue
            # Gravity model.
            tm[r, c] = egress[r] * ingress[c] / (ingress.sum() - ingress[r])

    return genProto(tor_level, num_clusters, num_nodes, tm)

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
                demand.volume_mbps = floor(TM[(i - 1) * 32 + u - 1,
                                              (j - 1) * 32 + v - 1])
        else:
            # Populate AggrBlock-level demand matrix.
            if i == j:
                continue
            demand = tm_proto.demands.add()
            demand.src = f'{NETNAME}-c{i}-ab1'
            demand.dst = f'{NETNAME}-c{j}-ab1'
            demand.volume_mbps = floor(TM[i-1, j-1])

    return tm_proto
