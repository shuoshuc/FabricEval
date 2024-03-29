import math

import gurobipy as gp
import numpy as np
import proto.te_solution_pb2 as TESolution
from google.protobuf import text_format
from gurobipy import GRB

import common.flags as FLAG
from common.common import PRINTV
from topology.topology import Topology, filterPathSetWithSeg
from traffic.traffic import Traffic


def prettyPrint(verbose, te_sol):
    '''
    Pretty prints the TE solution.
    '''
    if FLAG.VERBOSE >= verbose:
        print('\n===== TE solution starts =====')
        for c, sol in te_sol.items():
            # Raw solution has flows in Tbps, converts back to Mbps.
            print(f'Demand: [{c[0]}] => [{c[1]}], {c[2] * 1000000} Mbps')
            for path_name, flow in sol.items():
                print(f'    {flow} Mbps on {path_name}')
        print('===== TE solution ends =====\n')

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
            # Stores demand in Tbps instead of Mbps.
            self.commodity_idx_std[idx] = (s, t, d / 1000000)
            path_set = topo_obj.findPathSetOfAggrBlockPair(s, t)
            self.commodity_path_sets[idx] = path_set

    def solve(self):
        '''
        Constructs and then solves the MCF optimization.
        '''
        try:
            # Initialize a new model
            m = gp.Model("global_mcf")
            m.setParam("LogToConsole", 1 if FLAG.VERBOSE >= 2 else 0)
            #m.setParam("FeasibilityTol", 1e-9)
            #m.setParam("NodefileStart", 0.5)
            #m.setParam("NodefileDir", "/tmp")
            m.setParam("Threads", 0)
            #m.setParam("TimeLimit", 120)
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
                # Converts capacity from Mbps to Tbps.
                c[(s, t)] = link.capacity / 1000000
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
                # Hedging constraint below.
                if not FLAG.ENABLE_HEDGING:
                    continue
                # Spread S of 0 is illegal, should not add hedging constraint.
                if math.isclose(FLAG.S, 0.0, rel_tol=1e-10):
                    print(f'[ERROR] hedging spread {FLAG.S} cannot be 0!')
                    continue
                # Obtains the capacity of every path in the path_set.
                cp_list = [self._topo.findCapacityOfPathTuple(p) \
                           for p in path_set.keys()]
                # For each path in a commodity, the flow assigned to the path
                # cannot exceed a fraction S of the total bisection capacity.
                for k, p in enumerate(path_set.keys()):
                    m.addConstr(f[idx][p] <= demand *
                                cp_list[k] / (sum(cp_list) * FLAG.S))

            # Optimize model
            m.optimize()

            # Extracts and organizes final solution.
            te_sol_by_commodity = {}
            te_sol_by_src = {}
            for f in m.getVars():
                if 'f_' in f.VarName:
                    # Skips empty flows.
                    if f.X == 0.0:
                        continue
                    splits = f.VarName.split('_')
                    # Extracts commodity and path from variable name.
                    # Also converts flow from Tbps back to Mbps.
                    i, path, flow = int(splits[1]), splits[2], f.X * 1000000
                    te_sol_by_commodity.setdefault(self.commodity_idx_std[i],
                                                   {})[path] = flow
            PRINTV(1, f'Solver obj MLU: {m.ObjVal}')
            prettyPrint(1, te_sol_by_commodity)

            for (s, t, _), path_map in te_sol_by_commodity.items():
                # Allocates a new TEIntent for source node s.
                te_intent = te_sol_by_src.setdefault(s, TESolution.TEIntent())
                te_intent.target_block = s
                # Allocates a new PrefixIntent for destination node t.
                prefix_intent = te_intent.prefix_intents.add()
                prefix_intent.dst_name = t
                # PrefixType of this entry is always SRC. But there might be
                # TRANSIT type entries generated along the parsing, which will
                # be appended to other TEIntent.
                prefix_intent.type = TESolution.PrefixIntent.PrefixType.SRC
                # Converts flows on paths to flows on links.
                flow_dist = self._topo.distributeFlows(path_map)
                for (u, v), port_weight_map in flow_dist.items():
                    if u == s:
                        # Merge all source flows across single-segment paths
                        # into this prefix_intent.
                        for port, weight in port_weight_map.items():
                            nexthop_entry = prefix_intent.nexthop_entries.add()
                            nexthop_entry.nexthop_port = port
                            nexthop_entry.weight = weight
                    else:
                        # u != s means that we are dealing with transit flows.
                        # Find the corresponding TEIntent for source AggrBlock
                        # u and add them accordingly.
                        transit_te_intent = te_sol_by_src.setdefault(u,
                            TESolution.TEIntent())
                        transit_te_intent.target_block = u
                        # Allocates a new PrefixIntent for destination node v.
                        # Note that another PrefixIntent might exist, but that
                        # is for SRC, not TRANSIT.
                        prefix_intent_v = transit_te_intent.prefix_intents.add()
                        prefix_intent_v.dst_name = v
                        prefix_intent_v.type = \
                            TESolution.PrefixIntent.PrefixType.TRANSIT
                        # Populates `prefix_intent_v` with transit flows.
                        for port, weight in port_weight_map.items():
                            nexthop_entry = prefix_intent_v.nexthop_entries.add()
                            nexthop_entry.nexthop_port = port
                            nexthop_entry.weight = weight

            # Packs per-src TEIntent into a TESolution.
            sol = TESolution.TESolution()
            sol.type = self._traffic.getDemandType()
            for src, te_intent in te_sol_by_src.items():
                sol.te_intents.append(te_intent)
            return sol

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
            return []
        except AttributeError:
            print('Encountered an attribute error')
            return []
