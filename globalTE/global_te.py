import gurobipy as gp
import numpy as np
import proto.te_solution_pb2 as te_sol
from google.protobuf import text_format
from gurobipy import GRB

from topology.topology import Topology, filterPathSetWithSeg
from traffic.traffic import Traffic

# VERBOSE=0: no Gurobi log. VERBOSE=1: Gurobi final log only. VERBOSE=2: full
# Gubrobi log.
VERBOSE = 0

def prettyPrint(te_sol):
    '''
    Pretty prints the TE solution.
    '''
    print('===== TE solution =====')
    for c, sol in te_sol.items():
        print(f'Demand: [{c[0]}] => [{c[1]}], {c[2]} Mbps')
        for path_name, flow in sol.items():
            print(f'    {flow} Mbps on {path_name}')

class GlobalTE:
    '''
    Global TE solves a multi-commodity flow (MCF) load balancing problem on
    a network defined by input topology. The commodities are defined in the
    input traffic information. It generates the optimal TE solution as protobuf.
    '''
    def __init__(self, topo_obj, traffic_obj):
        self._topo = topo_obj
        self._traffic = traffic_obj
        # commodity path set map: integer index of commodity to its path set.
        self.commodity_path_sets = {}
        # Map from the integer index of a commodity to its (src, dst, demand).
        self.commodity_idx_std = {}
        for idx, ((s, t), d) in enumerate(traffic_obj.getAllDemands().items()):
            self.commodity_idx_std[idx] = (s, t, d)
            path_set = topo_obj.findPathSetOfAggrBlockPair(s, t)
            self.commodity_path_sets[idx] = path_set

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
            # A map from link's (x, y) to link utilization, u(x, y). Note that
            # the term 'link' in this class means abstract level path.
            u = {}
            # A map from link's (x, y) to link capacity, c(x, y).
            c = {}
            # Iterate over all paths in topo. Again, we call path 'link' here.
            for link_name, link in self._topo.getAllPaths().items():
                s, t = link.src_aggr_block.name, link.dst_aggr_block.name
                u[(s, t)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                     name="u_" + link_name)
                c[(s, t)] = link.capacity
            # fip is the amount of flow in commodity i assigned on path p.
            # f is a map of a map: {commodity index: {(s, m, t): fip}} 
            f = {}
            for i, path_set in self.commodity_path_sets.items():
                for path in path_set.keys():
                    f.setdefault(i, {})[path] = m.addVar(vtype=GRB.CONTINUOUS,
                        lb=0, name=f"f_{i}_{':'.join(path)}")

            # Step 2: set objective.
            m.setObjective(umax, GRB.MINIMIZE)

            # Step 3: add constraints.
            for link, u_link in u.items():
                # Definition of max link utilization.
                # For each link, u(x, y) <= umax.
                m.addConstr(u_link <= umax)

                # Definition of link utilization. fip_link contains all fip that
                # traverses `link`.
                fip_link = []
                for idx, path_set in self.commodity_path_sets.items():
                    # For each commodity, get the paths that contain `link`.
                    filtered_path_set = filterPathSetWithSeg(path_set, link)
                    # fip of paths that contain `link` will be summed up to
                    # compute u(x, y).
                    for path in filtered_path_set.keys():
                        fip_link.append(f[idx][path])
                # For each link, u(x, y) == sum_i(sum_Pi[x,y](fip) / c(x, y),
                # which is equivalent to u(x, y) == sum(fip_link) / c(x, y)
                m.addConstr(u_link == gp.quicksum(fip_link) / c[link])

                # Link capacity constraint.
                # For each link, sum(fip_link) <= c(x, y).
                m.addConstr(gp.quicksum(fip_link) <= c[link])

            # Setp 3 continued: flow conservation constraint for each commodity.
            for idx, path_set in self.commodity_path_sets.items():
                _, _, demand = self.commodity_idx_std[idx]
                # For each commodity i, sum_p(fip) == demand_i.
                m.addConstr(gp.quicksum(list(f[idx].values())) == demand)

            # Optimize model
            m.optimize()

            # Extracts and organizes final solution.
            te_sol = {}
            for f in m.getVars():
                if 'f_' in f.VarName:
                    splits = f.VarName.split('_')
                    i, path = int(splits[1]), splits[2]
                    te_sol.setdefault(self.commodity_idx_std[i], {})[path] = f.X

            return te_sol

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
    te_sol = global_te.solve()
    prettyPrint(te_sol)
