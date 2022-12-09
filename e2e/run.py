import proto.topology_pb2 as topo
import proto.traffic_pb2 as traffic_pb2
from google.protobuf import text_format

from topology.topogen import generateToy3
from topology.topology import Topology, filterPathSetWithSeg, loadTopo
from traffic.traffic import Traffic

if __name__ == "__main__":
    net_proto = generateToy3()
    print(text_format.MessageToString(net_proto))
