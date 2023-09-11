import itertools
from math import floor

import numpy as np
import proto.traffic_pb2 as traffic
from numpy.random import default_rng
from scipy.stats import bernoulli, truncexpon, uniform

import common.flags as FLAG


def tmgen(tor_level, cluster_vector, num_nodes, model, dist='exp', netname='',
          raw_tm=False):
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
    netname: name of the fabric.
    raw_tm: If True, returns the raw TM in Python list format instead of
            protobuf format.
    '''
    rng = default_rng()
    num_clusters = cluster_vector.size
    size = num_clusters * num_nodes if tor_level else num_clusters
    tm = np.zeros(shape=(size, size))
    if model == 'flat':
        # Generates a flat traffic demand matrix. If tor_level=True, each
        # src-dst pair sees 153 Mbps demand. If false, each src-dst pair sees
        # 80000 Mbps demand.
        tm[tm >= 0] = 30000 * 8 / (num_nodes * num_clusters - 1) if tor_level \
            else 20000 * 256 / (num_clusters - 1)
    elif model == 'uniform':
        # Generates a uniform random traffic demand matrix. Each src-dst pair
        # will not exceed the value of a same entry in the flat TM.
        upper_bound = 40000 * 8 / (num_nodes * num_clusters - 1) if tor_level \
            else 40000 * 256 / (num_clusters - 1)
        for r, c in np.ndindex(tm.shape):
            tm[r, c] = rng.uniform(low=0, high=upper_bound)
    elif model == 'single':
        # Send from the first node to the last node at a low load. There must
        # be at least 2 nodes in the TM.
        # Note: only generates 1 commodity, this is for debugging.
        tm[0, -1] = 40000 * 4 * 0.5
    elif model == 'gravity':
        # Generates a traffic demand matrix following the gravity model. Each
        # src-dst pair has a demand proportional to the product of their egress
        # and ingress demands. The block total ingress/egress volume is sampled
        # from a uniform random/exponential/Pareto distribution, as specified by
        # `dist`.
        egress, ingress = np.array([]), np.array([])
        egress = genTotalDemand(tor_level, cluster_vector, num_nodes, dist)
        # Set block total ingress to be the same as egress if flag is true.
        # Otherwise, sample another set of values from the same distribution.
        if FLAG.EQUAL_INGRESS_EGRESS:
            ingress = egress
        else:
            ingress = genTotalDemand(tor_level, cluster_vector, num_nodes, dist)
            # Rescale the ingress vector so that its sum equals egress. (Total
            # egress and ingress must match).
            ingress *= egress.sum() / ingress.sum()
        # Finds the number of latest generation blocks. We only set the latest
        # generation blocks to empty.
        num_latest = np.unique(cluster_vector, return_counts=True)[1][-1]
        # Designates P_SPARSE blocks to be empty. Stores their block indices.
        # Note that if a block is empty, its egress and ingress demands are both
        # set to 0, regardless of EQUAL_INGRESS_EGRESS.
        empty_blocks = np.nonzero(bernoulli.rvs(1 - FLAG.P_SPARSE,
                                                size=num_latest) == 0)[0] \
            + (len(cluster_vector) - num_latest)
        # `r` is row vector for src, `c` is column vector for dst.
        for r, c in np.ndindex(tm.shape):
            block_r = r // num_nodes if tor_level else r
            block_c = c // num_nodes if tor_level else c
            # If a block is designated empty, keep its row/column vector 0.
            if block_r in empty_blocks or block_c in empty_blocks:
                continue
            # Gravity model.
            tm[r, c] = egress[r] * ingress[c] / ingress.sum()

    return genProto(tor_level, num_clusters, num_nodes, tm, netname, raw_tm)

def genTotalDemand(tor_level, cluster_vector, num_nodes, dist, p_spike=0.1):
    '''
    Generates total ingress/egress demand for all end points.
    Returns a 1-D NumPy array.

    p_spike: probability to generate a spike (spike = 80% max capacity).
    '''
    rng = default_rng()
    # Step 1: Generates AggrBlock-level total demand.
    block_demand = np.array([])
    for i, f in enumerate(cluster_vector):
        upper_bound = 0
        for j, g in enumerate(cluster_vector):
            if i == j:
                continue
            upper_bound += 40000 * 4 * min(f, g)
        # With `p_spike` probability, generates a spike.
        upper_bound *= 0.8 if rng.uniform(low=0, high=1) < p_spike else 0.4
        scale = upper_bound / 2
        if dist == 'exp':
            X = truncexpon(b=upper_bound/scale, loc=0, scale=scale)
        elif dist == 'uniform':
            X = uniform(loc=0, scale=upper_bound)
        block_demand = np.concatenate((block_demand, X.rvs(1)))

    # Only needs block demand, job done.
    if not tor_level:
        return block_demand

    # Step 2: Generates ToR-level total demand.
    tor_demand = np.array([])
    for i in range(len(cluster_vector)):
        upper_bound = block_demand[i]
        scale = upper_bound / 2
        if dist == 'exp':
            X = truncexpon(b=upper_bound/scale, loc=0, scale=scale)
        elif dist == 'uniform':
            X = uniform(loc=0, scale=upper_bound)
        tors_in_block = X.rvs(num_nodes)
        # Rescales the tor vector so that it sums to upper_bound.
        tor_demand = np.concatenate((tor_demand,
            tors_in_block / tors_in_block.sum() * upper_bound))
    return tor_demand

def genProto(tor_level, num_clusters, num_nodes, TM, netname, raw_tm):
    '''
    Returns a traffic proto using the given traffic matrix `TM`.

    raw_tm: If True, constructs the return value in raw list format instead of
            protobuf format. Note that S1 nodes follow a slightly different
            naming scheme for convenience.
    '''
    tm_proto = traffic.TrafficDemand()
    tm_proto.type = traffic.TrafficDemand.DemandType.LEVEL_TOR if tor_level \
        else traffic.TrafficDemand.DemandType.LEVEL_AGGR_BLOCK

    # Each demand entry looks like:
    # [src node, src cluster id, dst node, dst cluster id, demand (Mbps)].
    rawTM = []

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
                if floor(TM[(i - 1) * num_nodes + u - 1,
                            (j - 1) * num_nodes + v - 1]) <= 0:
                    continue
                demand = tm_proto.demands.add()
                demand.src = f'{netname}-c{i}-ab1-s1i{u}'
                demand.dst = f'{netname}-c{j}-ab1-s1i{v}'
                demand.volume_mbps = floor(TM[(i - 1) * num_nodes + u - 1,
                                              (j - 1) * num_nodes + v - 1])
                if raw_tm:
                    rawTM.append([f'{netname}-c{i}-t{u}', f'{i}',
                                  f'{netname}-c{j}-t{v}', f'{j}',
                                  demand.volume_mbps])
        else:
            # Populate AggrBlock-level demand matrix.
            if i == j:
                continue
            # Skip zero entries for proto efficiency.
            if floor(TM[i-1, j-1]) <= 0:
                continue
            demand = tm_proto.demands.add()
            demand.src = f'{netname}-c{i}-ab1'
            demand.dst = f'{netname}-c{j}-ab1'
            demand.volume_mbps = floor(TM[i-1, j-1])
            if raw_tm:
                rawTM.append([demand.src, f'{i}', demand.dst, f'{j}',
                              demand.volume_mbps])

    return tm_proto if not raw_tm else rawTM
