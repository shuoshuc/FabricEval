import itertools
import math

import gurobipy as gp
from gurobipy import GRB

#import common.flags as FLAG
#from common.common import PRINTV


class StripingPlan:
    '''
    StripingPlan solves the problem of interconnecting clusters to form a
    network fabric. It assumes a full-mesh topology and tries to balance the
    links/paths assigned between any cluster pair as evenly as possible (unless
    explicitly requested otherwise).
    '''
    def __init__(self, num_clusters, cluster_radices):
        '''
        num_clusters: the total number of clusters to be connected.

        cluster_radices: a map from cluster id to the radix/degree (of egress
                         links). Note that links are bidirectional here.
                         Note that cluster id is 1-indexed.
        '''
        self.num_clusters = num_clusters
        self.cluster_radices = cluster_radices

    def _max_assignment(self, model):
        '''
        Builds an ILP formulation to maximize the number of links assigned
        between clusters subject to radix and balance constraints.

        model: pre-built empty model, needs decision vars and constraints.
        '''
        # Create variables: x[i][j] is the number of links assigned between
        # cluster i and j. x is a matrix of connectivity assignment.
        x = []
        for i in range(self.num_clusters):
            x_row = []
            for j in range(self.num_clusters):
                x_row.append(model.addVar(vtype=GRB.INTEGER, lb=0,
                                          ub=self.cluster_radices[i+1],
                                          name="x_{}_{}".format(i+1, j+1)))
            x.append(x_row)
        model.update()

        # Set objective: maximize total number of links assigned.
        model.setObjective(0.5 * gp.quicksum(model.getVars()), GRB.MAXIMIZE)

        # Add constraints.
        for i in range(self.num_clusters):
            # Add constraint: cluster radix bound.
            model.addConstr(gp.quicksum(x[i]) <= self.cluster_radices[i+1],
                            "cluster_radix_bound_{}".format(i+1))
            for j in range(self.num_clusters):
                if i == j:
                    # Add constraint: no self loops.
                    model.addConstr(x[i][i] == 0, "no_self_loop_{}".format(i+1))
                    continue
                # Add constraint: equal spread of links to peers.
                model.addConstr(x[i][j] >= math.floor(self.cluster_radices[i+1]\
                                                     / (self.num_clusters - 1)),
                                "even_spread_lb_{}_{}".format(i+1, j+1))
                model.addConstr(x[i][j] <= math.ceil(self.cluster_radices[i+1]\
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
            m = gp.Model("cluster_striping")
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

            sol_x = dict()
            for v in m.getVars():
                split = v.VarName.split('_')
                i, j = int(split[1]) - 1, int(split[2]) - 1
                sol_x[v.VarName] = v.X
            #PRINTV(2, 'Obj: %s' % m.ObjVal)
            #PRINTV(2, str(sol_x))
            print('Obj: %s' % m.ObjVal)
            print(str(sol_x))

            return None

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
            return []
        except AttributeError:
            print('Encountered an attribute error')
            return []

if __name__ == "__main__":
    cluster_radices = {
        1: 16,
        2: 16,
        3: 16,
        4: 16
    }
    sp = StripingPlan(4, cluster_radices)
    sp.solve()
