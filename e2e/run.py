import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from google.protobuf import text_format

import common.flags as FLAG
from globalTE.global_te import GlobalTE
from localTE.wcmp_alloc import WCMPAllocation
from topology.topogen import generateFabric
from topology.topology import Topology, filterPathSetWithSeg, loadTopo
from traffic.tmgen import tmgen
from traffic.traffic import Traffic

NETWORK = 'toy3'

TOY_TOPO = f'tests/data/{NETWORK}_topo.textproto'
TOY_TM = f'tests/data/{NETWORK}_traffic_gravity.textproto'
TOY_SOL = f'tests/data/{NETWORK}_te_sol.textproto'
# True to load topo from the above file.
LOAD_TOPO = False
# True to load TM from the above file.
LOAD_TM = True
# True to load TE solution from the above file.
LOAD_SOL = False

if __name__ == "__main__":
    # Initializes global flags before running the pipeline.
    if FLAG.GR_ALGO == 'eurosys':
        FLAG.EUROSYS_MOD = False
    elif FLAG.GR_ALGO == 'eurosys_mod':
        FLAG.EUROSYS_MOD = True
    elif FLAG.GR_ALGO == 'google':
        FLAG.IMPROVED_HEURISTIC = False
    elif FLAG.GR_ALGO == 'google_new':
        FLAG.IMPROVED_HEURISTIC = True
    elif FLAG.GR_ALGO == 'carving':
        FLAG.IMPROVED_HEURISTIC = True
        # Use single process if invoking Gurobi. Gurobi is able to use all CPU
        # cores, no need to multi-process, which adds extra overhead.
        FLAG.PARALLELISM = 1
    elif FLAG.GR_ALGO == 'gurobi':
        FLAG.IMPROVED_HEURISTIC = True
        FLAG.PARALLELISM = 1
    else:
        print(f'[ERROR] unknown group reduction algorithm {FLAG.GR_ALGO}.')

    logpath = Path(sys.argv[1] + f'/{FLAG.GR_ALGO}')
    logpath.mkdir(parents=True, exist_ok=True)

    # Generates topology.
    net_proto = None
    if not LOAD_TOPO:
        net_proto = generateFabric(NETWORK)
        with (logpath / 'topo.textproto').open('w') as topo:
            topo.write(text_format.MessageToString(net_proto))
    toy_topo = Topology(TOY_TOPO, net_proto)
    print(f'{datetime.now()} [Step 1] topology generated.', flush=True)

    # Generates TM.
    traffic_proto = None
    if not LOAD_TM:
        traffic_proto = tmgen(tor_level=True,
                              cluster_vector=np.array([1]*22 + [2.5]*22 + [5]*21),
                              num_nodes=32,
                              model='gravity',
                              dist='exp',
                              netname=NETWORK)
        with (logpath / 'TM.textproto').open('w') as tm:
            tm.write(text_format.MessageToString(traffic_proto))
    toy_traffic = Traffic(toy_topo, TOY_TM, traffic_proto)
    print(f'{datetime.now()} [Step 2] traffic demand generated.', flush=True)

    # Runs global TE.
    sol = None
    if not LOAD_SOL:
        global_te = GlobalTE(toy_topo, toy_traffic)
        sol = global_te.solve()
        with (logpath / 'te_sol.textproto').open('w') as te_sol:
            te_sol.write(text_format.MessageToString(sol))
    print(f'{datetime.now()} [Step 3] global TE solution generated.', flush=True)
    #print(text_format.MessageToString(sol))

    # Runs local TE.
    wcmp_alloc = WCMPAllocation(toy_topo, toy_traffic, TOY_SOL, sol)
    wcmp_alloc.run()
    print(f'{datetime.now()} [Step 4] local TE solution generated.', flush=True)

    # Dumps stats.
    real_LUs = toy_topo.dumpRealLinkUtil()
    ideal_LUs = toy_topo.dumpIdealLinkUtil()
    delta_LUs = {}
    for k, (u, dcn) in real_LUs.items():
        delta_LUs[k] = (u - ideal_LUs[k][0], dcn)

    with (logpath / 'LU.csv').open('w') as LU:
        writer = csv.writer(LU)
        writer.writerow(["link name", "dcn facing", "ideal LU", "real LU", "delta"])
        for k, (v, dcn) in dict(sorted(delta_LUs.items(), key=lambda x: x[1][0],
                                       reverse=True)).items():
            writer.writerow([k, f'{dcn}', f'{ideal_LUs[k][0]}',
                             f'{real_LUs[k][0]}', f'{v}'])
    print(f'{datetime.now()} [Step 5] dump link util to LU.csv', flush=True)

    ecmp_util = toy_topo.dumpECMPUtil()
    with (logpath / 'node_ecmp.csv').open('w') as ecmp:
        writer = csv.writer(ecmp)
        writer.writerow(["node name", "ECMP util", "# groups"])
        for k, (util, num_g) in ecmp_util.items():
            writer.writerow([k, f'{util}', f'{num_g}'])
    print(f'{datetime.now()} [Step 6] dump node table util to node_ecmp.csv',
          flush=True)

    demand_admit = toy_topo.dumpDemandAdmission()
    with (logpath / 'node_demand.csv').open('w') as demand:
        writer = csv.writer(demand)
        writer.writerow(["node name", "total demand", "total admit", "ratio"])
        for node, (tot_demand, tot_admit, f) in demand_admit.items():
            writer.writerow([node, tot_demand, tot_admit, f])
    print(f'{datetime.now()} [Step 7] dump node demand to node_demand.csv',
          flush=True)

    path_div = wcmp_alloc.dumpPathDiversityStats()
    with (logpath / 'path_diversity.csv').open('w') as path_diversity:
        writer = csv.writer(path_diversity)
        writer.writerow(["node", "gid", "total volume (Mbps)", "orig paths",
                         "reduced paths", "pruned volume (Mbps)"])
        for (node, gid, vol), [orig_p, reduced_p, pruned_vol] in path_div.items():
            writer.writerow([node, gid, vol, orig_p, reduced_p, pruned_vol])
    print(f'{datetime.now()} [Step 8] dump path diversity to path_diversity.csv',
          flush=True)

    if FLAG.DUMP_GROUPS:
        groups_by_node = wcmp_alloc.dumpOrigS3SrcGroups()
        for node, groups in groups_by_node.items():
            newpath = logpath / 'group_dump'
            newpath.mkdir(parents=True, exist_ok=True)
            with (newpath / f'orig_groups_{node}.csv').open('w') as gdump:
                writer = csv.writer(gdump)
                for G in groups:
                    writer.writerow(G)
        print(f'{datetime.now()} [Step 9] dump groups to group_dump/orig_groups_*.csv',
              flush=True)

    reduced_groups = wcmp_alloc.dumpReducedGroups()
    with (logpath / 'reduced_groups.csv').open('w') as g_dump:
        writer = csv.writer(g_dump)
        writer.writerow(["#group type", "src", "dst", "group"])
        for g_type, src, dst, reduced_w in reduced_groups:
            writer.writerow([g_type, src, dst] + reduced_w)
    print(f'{datetime.now()} [Step 10] dump reduced groups to reduced_groups.csv',
          flush=True)
