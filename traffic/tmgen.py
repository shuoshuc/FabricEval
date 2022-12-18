import itertools
from math import floor

import numpy as np
import proto.traffic_pb2 as traffic
from numpy.random import default_rng
from scipy.stats import truncexpon, uniform

NETNAME = 'toy3'
# True means the block total ingress should equal its total egress.
EQUAL_INGRESS_EGRESS = True

def tmgen(tor_level, cluster_vector, num_nodes, model, dist='exp'):
    '''
    Generates a traffic demand matrix according to `model`.

    Returns the populated traffic proto.

    tor_level: Boolean. False means AggrBlock-level demand.
    cluster_vector: NumPy vector of the scale factor (aka, relative speed). For
                    example, a total of 2 40G clusters and 2 100G clusters would
                    look like: array([1, 1, 2.5, 2.5]).
    num_nodes: number of S1 nodes per cluster. Only used when tor_level=True.
    model: the type of TM to use, can be flat/uniform/gravity.
    dist: what distribution to use for sampling ingress/egress total demand, can
          be exp/uniform/pareto.
    '''
    rng = default_rng()
    num_clusters = cluster_vector.size
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
        # from a uniform random/exponential/Pareto distribution, as specified by
        # `dist`.
        egress, ingress = np.array([]), np.array([])
        factors, counts = np.unique(cluster_vector, return_counts=True)
        for f, count in zip(factors, counts):
            upper_bound = (40000 * 8 if tor_level else 40000 * 256) * 0.7 * f
            scale = upper_bound / 2
            if dist == 'exp':
                X = truncexpon(b=upper_bound/scale, loc=0, scale=scale)
            elif dist == 'uniform':
                X = uniform(loc=0, scale=upper_bound)
            # Sample egress total volume for all src of the same speed factor.
            sample_size = count * num_nodes if tor_level else count
            egress = np.concatenate((egress, X.rvs(sample_size)))
        # Set block total ingress to be the same as egress if flag is true.
        # Otherwise, sample another set of values from the same distribution.
        if EQUAL_INGRESS_EGRESS:
            ingress = egress
        else:
            for f, count in zip(factors, counts):
                up_bound = (40000 * 8 if tor_level else 40000 * 256) * 0.7 * f
                scale = up_bound / 2
                if dist == 'exp':
                    X = truncexpon(b=upper_bound/scale, loc=0, scale=scale)
                elif dist == 'uniform':
                    X = uniform(loc=0, scale=upper_bound)
                sample_size = count * num_nodes if tor_level else count
                ingress = np.concatenate((ingress, X.rvs(sample_size)))
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
                # Skip zero entries for proto efficiency.
                if floor(TM[(i - 1) * 32 + u - 1, (j - 1) * 32 + v - 1]) <= 0:
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
            # Skip zero entries for proto efficiency.
            if floor(TM[i-1, j-1]) <= 0:
                continue
            demand = tm_proto.demands.add()
            demand.src = f'{NETNAME}-c{i}-ab1'
            demand.dst = f'{NETNAME}-c{j}-ab1'
            demand.volume_mbps = floor(TM[i-1, j-1])

    return tm_proto
