import csv
import sys

import numpy as np
from google.protobuf import text_format

from globalTE.global_te import GlobalTE
from localTE.wcmp_alloc import WCMPAllocation
from topology.topogen import generateToy3
from topology.topology import Topology, filterPathSetWithSeg, loadTopo
from traffic.tmgen import tmgen
from traffic.traffic import Traffic

if __name__ == "__main__":
    logpath = sys.argv[1]
    net_proto = generateToy3()
    toy3 = Topology('', net_proto)
    print('[Step 1] topology generated.')
    #print(text_format.MessageToString(net_proto))
    traffic_proto = tmgen(tor_level=True,
                          cluster_vector=np.array([1]*22 + [2.5]*22 + [5]*21),
                          num_nodes=32,
                          model='gravity',
                          dist='exp',
                          netname='toy3')
    with open(f'{logpath}/TM.textproto', 'w') as tm:
        tm.write(text_format.MessageToString(traffic_proto))
    toy3_traffic = Traffic(toy3, '', traffic_proto)
    print('[Step 2] traffic demand generated.')
    global_te = GlobalTE(toy3, toy3_traffic)
    sol = global_te.solve()
    with open(f'{logpath}/te_sol.textproto', 'w') as te_sol:
        te_sol.write(text_format.MessageToString(sol))
    print('[Step 3] global TE solution generated.')
    #print(text_format.MessageToString(sol))
    wcmp_alloc = WCMPAllocation(toy3, '', sol)
    wcmp_alloc.run()
    print('[Step 4] local TE solution generated.')
    real_LUs = toy3.dumpRealLinkUtil()
    ideal_LUs = toy3.dumpIdealLinkUtil()
    delta_LUs = {}
    for k, v in real_LUs.items():
        delta_LUs[k] = v - ideal_LUs[k]

    print(f'[Step 5] dumping link util to {sys.argv[1]}')
    with open(f'{logpath}/LU.csv', 'w') as LU:
        writer = csv.writer(LU)
        writer.writerow(["link name", "ideal LU", "real LU", "delta"])
        for k, v in dict(sorted(delta_LUs.items(), key=lambda x: x[1],
                                reverse=True)).items():
            writer.writerow([k, f'{ideal_LUs[k]}', f'{real_LUs[k]}', f'{v}'])
