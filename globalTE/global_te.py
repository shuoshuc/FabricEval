from google.protobuf import text_format
from topology import Topology
from traffic import Traffic
import proto.te_solution_pb2 as te_sol
import gurobipy as gp
import numpy as np
from gurobipy import GRB

# VERBOSE=0: no Gurobi log. VERBOSE=1: Gurobi final log only. VERBOSE=2: full
# Gubrobi log.
VERBOSE = 0

class GlobalTE:
    '''
    Global TE solves a multi-commodity flow (MCF) load balancing problem on
    a network defined by input topology. The commodities are defined in the
    input traffic information. It generates the optimal TE solution as protobuf.
    '''
    def __init__(self, topo_obj, traffic_obj):
        self._topo = topo_obj
        self._traffic = traffic_obj

    def solve(self):
        '''
        Constructs and then solves the MCF optimization.
        '''
        try:
            # Initialize a new model
            m = gp.Model("global_mcf")
            m.setParam("LogToConsole", 1 if VERBOSE >= 2 else 0)
            m.setParam("FeasibilityTol", 1e-7)
            m.setParam("IntFeasTol", 1e-8)
            m.setParam("MIPGap", 1e-4)
            #m.setParam("NodefileStart", 0.5)
            m.setParam("NodefileDir", "/tmp")
            m.setParam("Threads", 0)
            m.setParam("TimeLimit", 120)
            #m.setParam("LogFile", "gurobi.log")

            # Construct model
            umax = m.addVar(vtype=GRB.CONTINUOUS, name="umax")

            # Optimize model
            m.optimize()

            # Extracts and organizes final solution.

            # TODO: return

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
            return []
        except AttributeError:
            print('Encountered an attribute error')
            return []

if __name__ == "__main__":
    pass
