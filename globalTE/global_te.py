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
            # A map from link name to link utilization, u(x, y). Note that the
            # term 'link' in this class means abstract level path.
            u = {}
            # A map from link name to link capacity, c(x, y).
            c = {}
            # fip is the amount of flow in commodity i assigned on path p.
            # f is a map of a map: {[path]: {[commodity]: fi(x, y)}} 
            f = {}
            # Outer loop: iterate over all paths.
            for path_name, path in self._topo.getAllPaths().items():
                u[path_name] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                        name="u_" + path_name)
                c[path_name] = path.capacity
                # Inner loop: iterate over all commodities.
                for i, (src, dst) in self.commodity_idx_st.items():
                    f.setdefault(path_name, {})[i] = m.addVar(
                        vtype=GRB.CONTINUOUS, lb=0, name=f"f_{i}_{path_name}")

            # Step 2: set objective.
            m.setObjective(umax, GRB.MINIMIZE)

            # Step 3: add constraints.
            for path_name, u_path in u.items():
                # Definition of max link utilization.
                # For each path, u(x, y) <= umax.
                m.addConstr(u_path <= umax)
                # Definition of link utilization.
                # For each path, u(x, y) == sum_i(fi[(x, y)]) / c(x, y).
                m.addConstr(u_path == gp.quicksum(list(f[path_name].values())) /
                            c[path_name])
                # Link capacity constraint.
                # For each path, sum_i(fi[(x, y)]) <= c(x, y).
                m.addConstr(gp.quicksum(list(f[path_name].values())) <=
                            c[path_name])

            # Setp 3 continued: flow conservation constraints for commodity i.
            for idx, (src, dst) in self.commodity_idx_st.items():
                demand = self._traffic.getDemand(src, dst)

                # Construct flow conservation constraint for src si.
                u_si_y, u_y_si = [], []
                # `u_si_y` is a list of link util vars from src of commodity i.
                for orig_path in self._topo.findOrigPathsOfAggrBlock(src):
                    u_si_y.append(f[orig_path][idx])
                # `u_y_si` is a list of link util vars to src of commodity i.
                for term_path in self._topo.findTermPathsOfAggrBlock(src):
                    u_y_si.append(f[term_path][idx])
                # Constraint: sum_y(fi[(si, y)]) - sum_y(fi(y, si)) = di
                m.addConstr(gp.quicksum(u_si_y) - gp.quicksum(u_y_si) ==
                            demand, "flow_conservation_src_{}".format(idx))

                # Construct flow conservation constraint for dst ti.
                u_x_ti, u_ti_x = [], []
                # `u_x_ti` is a list of link util vars to dst of commodity i.
                for term_path in self._topo.findTermPathsOfAggrBlock(dst):
                    u_x_ti.append(f[term_path][idx])
                # `u_ti_x` is a list of link util vars from dst of commodity i.
                for orig_path in self._topo.findOrigPathsOfAggrBlock(dst):
                    u_ti_x.append(f[orig_path][idx])
                # Constraint: sum_x(fi[(x, ti)]) - sum_x(fi(ti, x)) = di
                m.addConstr(gp.quicksum(u_x_ti) - gp.quicksum(u_ti_x) ==
                            demand, "flow_conservation_dst_{}".format(idx))

                # Construct flow conservation constraint for transit nodes.
                for aggr_block_name in self._topo.getAllAggrBlocks().keys():
                    if aggr_block_name in non_src_dst_nodes:
                        pass

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
