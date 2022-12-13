import proto.topology_pb2 as topo
import proto.traffic_pb2 as traffic_pb2
from google.protobuf import text_format

from globalTE.global_te import GlobalTE
from topology.topogen import generateToy3
from topology.topology import Topology, filterPathSetWithSeg, loadTopo
from traffic.tmgen import tmgen
from traffic.traffic import Traffic

if __name__ == "__main__":
    net_proto = generateToy3()
    toy3 = Topology('', net_proto)
    #print(text_format.MessageToString(net_proto))
    traffic_proto = tmgen(False, 65, 32, 'gravity')
    toy3_traffic = Traffic('', traffic_proto)
    global_te = GlobalTE(toy3, toy3_traffic)
    sol = global_te.solve()
    #print(text_format.MessageToString(sol))
    LUs = toy3.dumpLinkUtil()
    for k, v in LUs.items():
        print(f'{k}: {v}')
