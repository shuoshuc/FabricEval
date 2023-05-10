import itertools
import math
import random

import gurobipy as gp
import numpy as np
from gurobipy import GRB

#import common.flags as FLAG
#from common.common import PRINTV

# Number of S3 switches per cluster.
NUM_S3 = 4

def matrixIndexToSwitchIndex(mi):
    '''
    Converts the index in the striping matrix to the corresponding cluster index
    and S3 switch index.
    '''
    return mi // NUM_S3, mi % NUM_S3

def switchIndexToMatrixIndex(ci, si):
    '''
    Converts the cluster index and S3 switch index to the corresponding index
    in the striping matrix.
    '''
    return ci * NUM_S3 + si

class StripingPlan:
    '''
    StripingPlan solves the problem of interconnecting clusters to form a
    network fabric. It assumes a full-mesh topology and tries to balance the
    links/paths assigned between any cluster pair as evenly as possible (unless
    explicitly requested otherwise).
    '''
    def __init__(self, net_name, num_clusters, cluster_radices):
        '''
        net_name: name of the network, used for constructing physical striping
                  plan.

        num_clusters: the total number of clusters to be connected.

        cluster_radices: a map from cluster id to the radix/degree (of egress
                         links). Note that links are bidirectional here.
                         Note that cluster id is 1-indexed.
        '''
        self.net_name = net_name
        self.num_clusters = num_clusters
        self.cluster_radices = cluster_radices
        # A map from globally unique S3 index to its radix.
        self.s3_radices = {}
        # Distribute the cluster radix to each S3 switch. If the radix is
        # not a multiple of NUM_S3, the residual links will be round robin
        # assigned to the S3 switches of the cluster.
        for ci in range(self.num_clusters):
            for si in range(NUM_S3):
                mi = switchIndexToMatrixIndex(ci, si)
                self.s3_radices[mi] = self.cluster_radices[ci+1] // NUM_S3
            for si in range(self.cluster_radices[ci+1] \
                            - self.cluster_radices[ci+1] // NUM_S3 * NUM_S3):
                mi = switchIndexToMatrixIndex(ci, si)
                self.s3_radices[mi] += 1

    def _max_assignment(self, model):
        '''
        Builds an ILP formulation to maximize the number of links assigned
        between S3 switches subject to radix and balance constraints.

        model: pre-built empty model, needs decision vars and constraints.
        '''
        min_radix = min(self.s3_radices.values())

        # Create variables: x[i][j] is the number of links assigned between
        # S3 switch i and j. x is a matrix of connectivity assignment.
        x = []
        for i in range(self.num_clusters * NUM_S3):
            x_row = []
            for j in range(self.num_clusters * NUM_S3):
                x_row.append(model.addVar(vtype=GRB.INTEGER, lb=0,
                                          ub=self.s3_radices[i],
                                          name="x_{}_{}".format(i+1, j+1)))
            x.append(x_row)
        model.update()

        # Set objective: maximize total number of links assigned.
        model.setObjective(0.5 * gp.quicksum(model.getVars()), GRB.MAXIMIZE)

        # Add constraints.
        for i in range(self.num_clusters * NUM_S3):
            # Add constraint: S3 radix bound.
            model.addConstr(gp.quicksum(x[i]) <= self.s3_radices[i],
                            "s3_radix_bound_{}".format(i+1))
            for j in range(self.num_clusters * NUM_S3):
                if i == j:
                    # Add constraint: no self loops.
                    model.addConstr(x[i][i] == 0, "no_self_loop_{}".format(i+1))
                    continue
                _, si = matrixIndexToSwitchIndex(i)
                _, sj = matrixIndexToSwitchIndex(j)
                if si != sj:
                    # Add constraint: no cross connect between S3 switches of
                    # different locations (intra-cluster indices).
                    model.addConstr(x[i][j] == 0,
                                    "no_x_connect_{}_{}".format(i+1, j+1))
                    continue
                # Add constraint: equal spread of links to peers.
                # Note: lower bound should be the S3 switch of min radix, since
                # we allow ports to be idle.
                model.addConstr(x[i][j] >= math.floor(min_radix \
                                                     / (self.num_clusters - 1)),
                                "even_spread_lb_{}_{}".format(i+1, j+1))
                model.addConstr(x[i][j] <= math.ceil(self.s3_radices[i] \
                                                     / (self.num_clusters - 1)),
                                "even_spread_ub_{}_{}".format(i+1, j+1))
                # Matrix x is symmetric, we can save half of the constraints.
                if i < j:
                    # Add constraint: bidi connections - x_ij == x_ji.
                    model.addConstr(x[i][j] == x[j][i],
                                    "bidi_connection_{}_{}".format(i+1, j+1))

        return model

    def solve(self):
        try:
            # Initialize a new model
            m = gp.Model("striping")
            #m.setParam("LogToConsole", 1 if FLAG.VERBOSE >= 2 else 0)
            m.setParam("LogToConsole", 1)
            m.setParam("FeasibilityTol", 1e-7)
            m.setParam("IntFeasTol", 1e-8)
            m.setParam("MIPGap", 1e-4)
            #m.setParam("NodefileStart", 0.5)
            m.setParam("NodefileDir", "/tmp")
            m.setParam("Threads", 0)
            #m.setParam("TimeLimit", FLAG.GUROBI_TIMEOUT)
            #m.setParam("LogFile", "gurobi.log")

            # Construct model
            m.setParam("NonConvex", 2)
            m.setParam("MIPFocus", 2)
            m = self._max_assignment(m)

            # Optimize model
            m.optimize()

            #PRINTV(2, 'Obj: %s' % m.ObjVal)
            print(f'[Obj] max links assigned in DCN: {m.ObjVal}')

            # A striping matrix for all S3 switches.
            mat_s3 = np.zeros(shape=(self.num_clusters * NUM_S3,
                                     self.num_clusters * NUM_S3))
            for v in m.getVars():
                tot_links = int(v.X)
                if not tot_links:
                    continue
                split = v.VarName.split('_')
                # Extract the two S3 switch indices.
                i, j = int(split[1]) - 1, int(split[2]) - 1
                mat_s3[i][j] = tot_links

            # A map from globally unique S3 switch id to a list of ports
            # available for establishing connections.
            available_ports = {}
            # A list of tuples, each tuple consists of 2 ports that can form a
            # bidi link.
            port_pairs = []
            # Iterate through the S3 radix map and populate a list of available
            # ports for the connections.
            for mi, radix in self.s3_radices.items():
                ci, si = matrixIndexToSwitchIndex(mi)
                for pi in range(radix):
                    available_ports.setdefault(mi, []).append(
                        f'{self.net_name}-c{ci+1}-ab1-s3i{si+1}-p{2*pi+1}')

            # Iterate through the striping matrix to construct port pairs.
            for mi, row in enumerate(mat_s3):
                ci, si = matrixIndexToSwitchIndex(mi)
                for mj, tot_links in enumerate(row):
                    # The striping matrix is symmetric, we only need to process
                    # half of it.
                    if not tot_links or mi >= mj:
                        continue
                    cj, sj = matrixIndexToSwitchIndex(mj)
                    for _ in range(int(tot_links)):
                        p1 = available_ports[mi].pop(0)
                        p2 = available_ports[mj].pop(0)
                        port_pairs.append((p1, p2))

            return port_pairs

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
            return []
        except AttributeError:
            print('Encountered an attribute error')
            return []

if __name__ == "__main__":
    '''
    cluster_radices = {
        1: 16,
        2: 16,
        3: 8,
        4: 16
    }
    sp = StripingPlan(4, cluster_radices)
    '''
    cluster_radices = {
        1: 8,
        2: 16,
        3: 16,
    }
    sp = StripingPlan('toy', 3, cluster_radices)
    port_pairs = sp.solve()
    print(port_pairs)
