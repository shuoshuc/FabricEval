import numpy as np
import proto.topology_pb2 as topo
import proto.traffic_pb2 as traffic_pb2
from google.protobuf import text_format

from globalTE.global_te import GlobalTE
from localTE.wcmp_alloc import WCMPAllocation
from topology.topogen import generateToy3
from topology.topology import Topology, filterPathSetWithSeg, loadTopo
from traffic.tmgen import tmgen
from traffic.traffic import Traffic

TOY3_TE_SOL_PATH = 'tests/data/toy3_te_sol.textproto'

if __name__ == "__main__":
    net_proto = generateToy3()
    toy3 = Topology('', net_proto)
    print('[Step 1] topology generated.')
    #print(text_format.MessageToString(net_proto))
    traffic_proto = tmgen(tor_level=True,
                          cluster_vector=np.array([1]*22 + [2.5]*22 + [5]*21),
                          num_nodes=32,
                          model='gravity',
                          dist='exp')
    toy3_traffic = Traffic(toy3, '', traffic_proto)
    print('[Step 2] traffic demand generated.')
    global_te = GlobalTE(toy3, toy3_traffic)
    sol = global_te.solve()
    print('[Step 3] global TE solution generated.')
    #print(text_format.MessageToString(sol))
    wcmp_alloc = WCMPAllocation(toy3, '', sol)
    wcmp_alloc.run()
    print('[Step 4] local TE solution generated.')
    real_LUs = toy3.dumpRealLinkUtil()
    ideal_LUs = toy3.dumpIdealLinkUtil()
    for k, v in real_LUs.items():
        print(f'{k}: {ideal_LUs[k]} => {v}')
