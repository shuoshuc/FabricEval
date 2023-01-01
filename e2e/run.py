import csv
import sys
from datetime import datetime

import numpy as np
from google.protobuf import text_format

from globalTE.global_te import GlobalTE
from localTE.wcmp_alloc import WCMPAllocation
from topology.topogen import generateToy3
from topology.topology import Topology, filterPathSetWithSeg, loadTopo
from traffic.tmgen import tmgen
from traffic.traffic import Traffic

TOY3_SOL = 'tests/data/te_sol.textproto'

if __name__ == "__main__":
    logpath = sys.argv[1]
    net_proto = generateToy3()
    toy3 = Topology('', net_proto)
    print(f'{datetime.now()} [Step 1] topology generated.', flush=True)
    #print(text_format.MessageToString(net_proto))
    traffic_proto = tmgen(tor_level=False,
                          cluster_vector=np.array([1]*22 + [2.5]*22 + [5]*21),
                          num_nodes=32,
                          model='gravity',
                          dist='exp',
                          netname='toy3')
    with open(f'{logpath}/TM.textproto', 'w') as tm:
        tm.write(text_format.MessageToString(traffic_proto))
    toy3_traffic = Traffic(toy3, '', traffic_proto)
    print(f'{datetime.now()} [Step 2] traffic demand generated.', flush=True)
    global_te = GlobalTE(toy3, toy3_traffic)
    sol = global_te.solve()
    with open(f'{logpath}/te_sol.textproto', 'w') as te_sol:
        te_sol.write(text_format.MessageToString(sol))
    print(f'{datetime.now()} [Step 3] global TE solution generated.', flush=True)
    #print(text_format.MessageToString(sol))
    wcmp_alloc = WCMPAllocation(toy3, '', sol)
    wcmp_alloc.run()
    print(f'{datetime.now()} [Step 4] local TE solution generated.', flush=True)
    real_LUs = toy3.dumpRealLinkUtil()
    ideal_LUs = toy3.dumpIdealLinkUtil()
    delta_LUs = {}
    for k, v in real_LUs.items():
        delta_LUs[k] = v - ideal_LUs[k]

    print(f'{datetime.now()} [Step 5] dump link util to LU.csv', flush=True)
    with open(f'{logpath}/LU.csv', 'w') as LU:
        writer = csv.writer(LU)
        writer.writerow(["link name", "ideal LU", "real LU", "delta"])
        for k, v in dict(sorted(delta_LUs.items(), key=lambda x: x[1],
                                reverse=True)).items():
            writer.writerow([k, f'{ideal_LUs[k]}', f'{real_LUs[k]}', f'{v}'])

    print(f'{datetime.now()} [Step 6] dump node table util to node_ecmp.csv',
          flush=True)
    ecmp_util = toy3.dumpECMPUtil()
    with open(f'{logpath}/node_ecmp.csv', 'w') as ecmp:
        writer = csv.writer(ecmp)
        writer.writerow(["node name", "ECMP util", "# groups"])
        for k, (util, num_g) in ecmp_util.items():
            writer.writerow([k, f'{util}', f'{num_g}'])

    print(f'{datetime.now()} [Step 7] dump node demand to node_demand.csv',
          flush=True)
    demand_admit = toy3.dumpDemandAdmission()
    with open(f'{logpath}/node_demand.csv', 'w') as demand:
        writer = csv.writer(demand)
        writer.writerow(["node name", "total demand", "total admit", "ratio"])
        for node, (tot_demand, tot_admit, f) in demand_admit.items():
            writer.writerow([node, tot_demand, tot_admit, f])
