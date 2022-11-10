from google.protobuf import text_format
from topology.topology import Topology
from traffic.traffic import Traffic
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
            # umax for maximum link utilization.
            umax = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="umax")
            # A map from path name to path utilization.
            u = {}
            # fi(x, y) is the amount of flow in commodity i assigned on (x, y).
            # f is a map of a map: {[path]: {[commodity]: fi(x, y)}} 
            f = {}
            for path_name, path in self._topo.getAllPaths().items():
                u[path_name] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                        name="u_" + path_name)
                for (src, dst), vol in self._traffic.getDemand().items():
                    f[path_name]

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
    TOY2_PATH = 'tests/data/toy2.textproto'
    TOY2_TRAFFIC_PATH = 'tests/data/toy2_traffic.textproto'
    toy2 = Topology(TOY2_PATH)
    toy2_traffic = Traffic(TOY2_TRAFFIC_PATH)
    global_te = GlobalTE(toy2, toy2_traffic)
    global_te.solve()
