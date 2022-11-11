import gurobipy as gp
import numpy as np
import proto.te_solution_pb2 as te_sol
from google.protobuf import text_format
from gurobipy import GRB

from topology.topology import Topology
from traffic.traffic import Traffic

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
        # Auxiliary map from an integer index to (src, dst).
        self.commodity_idx_st = {}
        for idx, ((s, t), _) in enumerate(traffic_obj.getAllDemands().items()):
            self.commodity_idx_st[idx] = (s, t)

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

            # Step 1: create decision variables.
            # umax for maximum link utilization.
            umax = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="umax")
            # A map from path name to path utilization.
            u = {}
            # A map from path name to path capacity.
            c = {}
            # fi(x, y) is the amount of flow in commodity i assigned on (x, y).
            # f is a map of a map: {[path]: {[commodity]: fi(x, y)}} 
            f = {}
            for path_name, path in self._topo.getAllPaths().items():
                u[path_name] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                        name="u_" + path_name)
                # f_map = f.setdefault(path_name, {})
                c[path_name] = path.capacity
                for (src, dst), _ in self._traffic.getAllDemands().items():
                    f.setdefault(path_name, {})[(src, dst)] = m.addVar(
                        vtype=GRB.CONTINUOUS, name=f"f_{src}:{dst}_{path_name}")
                    '''
                    f_map[(src, dst)] = m.addVar(vtype=GRB.CONTINUOUS,
                                                 name=f"f_{src}:{dst}_{path_name}")
                    '''

            # Step 2: set objective.
            m.setObjective(umax, GRB.MINIMIZE)

            # Step 3: add constraints.
            for path_name, u_path in u.items():
                # For each path, u(x, y) <= umax.
                m.addConstr(u_path <= umax)
                # For each path, u(x, y) == sum(fi[(x, y)]) / c(x, y).
                m.addConstr(u_path == gp.quicksum(list(f[path_name].values())) /
                            c[path_name])
                # For each path, sum(fi[(x, y)]) <= c(x, y). (Path capacity)
                m.addConstr(gp.quicksum(list(f[path_name].values())) <=
                            c[path_name])

            for idx, (src, dst) in self.commodity_idx_st.items():
                demand = self._traffic.getDemand(src, dst)
                # Construct flow conservation constraint for src.
                u_si_y, u_y_si = [], []
                # A list of u for fi(si, y).
                for orig_path in self._topo.findOrigPathsOfAggrBlock(src):
                    u_si_y.append(f[orig_path][(src, dst)])
                # A list of u for fi(y, si).
                for term_path in self._topo.findTermPathsOfAggrBlock(src):
                    u_y_si.append(f[term_path][(src, dst)])
                # sum(fi[(si, y)]) - sum(fi(y, si)) = di
                m.addConstr(gp.quicksum(u_si_y) - gp.quicksum(u_y_si) ==
                            demand, "flow_conservation_src_{}".format(idx))

                # Construct flow conservation constraint for dst.
                u_x_ti, u_ti_x = [], []
                # A list of u for fi(x, ti).
                for term_path in self._topo.findTermPathsOfAggrBlock(dst):
                    u_x_ti.append(f[term_path][(src, dst)])
                # A list of u for fi(ti, x).
                for orig_path in self._topo.findOrigPathsOfAggrBlock(dst):
                    u_ti_x.append(f[orig_path][(src, dst)])
                # sum(fi[(x, ti)]) - sum(fi(ti, x)) = di
                m.addConstr(gp.quicksum(u_x_ti) - gp.quicksum(u_ti_x) ==
                            demand, "flow_conservation_dst_{}".format(idx))

                # Construct flow conservation constraint for transit nodes.
                # TODO:

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
