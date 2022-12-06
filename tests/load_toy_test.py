import ipaddress
import unittest

import proto.traffic_pb2 as traffic_pb2

from topology.topology import Topology, filterPathSetWithSeg, loadTopo
from traffic.traffic import Traffic, loadTraffic

P1 = 'toy1-c1-ab1-s2i1-p1'
P2 = 'toy1-c1-ab1-s2i1-p2'
P3 = 'toy1-c1-ab1-s2i2-p1'
P4 = 'toy1-c1-ab1-s2i2-p2'
P5 = 'toy1-c1-ab1-s3i1-p1'
P6 = 'toy1-c1-ab1-s3i1-p2'
P7 = 'toy1-c1-ab1-s3i2-p1'
P8 = 'toy1-c1-ab1-s3i2-p2'
# Links in toy1: P1-P5, P2-P7, P3-P6, P4-P8.
TOY1_PATH = 'tests/data/toy1.textproto'
P9 = 'toy2-c3-ab1-s1i1-p1'
P10 = 'toy2-c3-ab1-s2i1-p1'
P11 = 'toy2-c1-ab1-s3i2-p4'
P12 = 'toy2-c3-ab1-s3i2-p3'
PATH1 = 'toy2-c1-ab1:toy2-c2-ab1'
PATH2 = 'toy2-c2-ab1:toy2-c1-ab1'
LINK1 = 'toy2-c1-ab1-s3i1-p3:toy2-c2-ab1-s3i1-p3'
LINK2 = 'toy2-c1-ab1-s3i2-p3:toy2-c2-ab1-s3i2-p3'
C1AB1 = 'toy2-c1-ab1'
C2AB1 = 'toy2-c2-ab1'
C3AB1 = 'toy2-c3-ab1'
TOY2_PATH = 'tests/data/toy2.textproto'
TOY2_TRAFFIC_PATH = 'tests/data/toy2_traffic.textproto'
TOR1 = 'toy2-c1-ab1-s1i1'
TOR2 = 'toy2-c3-ab1-s1i4'

class TestLoadToyNet(unittest.TestCase):
    def test_load_invalid_topo(self):
        self.assertEqual(None, loadTopo(''))

    def test_load_invalid_traffic(self):
        self.assertEqual(None, loadTraffic(''))

    def test_load_valid_toynet(self):
        toy1 = loadTopo(TOY1_PATH)
        self.assertNotEqual(None, toy1)

    def test_toy1_entities(self):
        toy1 = loadTopo(TOY1_PATH)
        # check network
        self.assertEqual('toy1', toy1.name)
        self.assertEqual(1, len(toy1.clusters))
        self.assertEqual(0, len(toy1.paths))
        self.assertEqual(8, len(toy1.links))
        # check cluster
        cluster = toy1.clusters[0]
        self.assertEqual('toy1-c1', cluster.name)
        self.assertEqual(1, len(cluster.aggr_blocks))
        # check aggr block
        aggr_block = cluster.aggr_blocks[0]
        self.assertEqual('toy1-c1-ab1', aggr_block.name)
        self.assertEqual(4, len(aggr_block.nodes))
        # check all nodes
        for node in aggr_block.nodes:
            self.assertEqual(2, len(node.ports))
            self.assertEqual(4000, node.flow_limit)
            self.assertEqual(4000, node.ecmp_limit)
            self.assertEqual(128, node.group_limit)
        # check node s2i1
        s2i1 = aggr_block.nodes[0]
        self.assertEqual('toy1-c1-ab1-s2i1', s2i1.name)
        self.assertEqual(2, s2i1.stage)
        self.assertEqual(1, s2i1.index)
        # check all ports
        for port in s2i1.ports:
            self.assertEqual(100 * 1000, port.port_speed_mbps)
            self.assertEqual(False, port.dcn_facing)
        # check port s2i1-p1
        port1 = s2i1.ports[0]
        self.assertEqual('toy1-c1-ab1-s2i1-p1', port1.name)
        # check 1st link
        link = toy1.links[0]
        self.assertEqual(P1 + ':' + P5, link.name)
        self.assertEqual(P1, link.src_port_id)
        self.assertEqual(P5, link.dst_port_id)
        self.assertEqual(100 * 1000, link.link_speed_mbps)

    def test_toy2_topology_construction(self):
        toy2 = Topology(TOY2_PATH)
        self.assertEqual(3, toy2.numClusters())
        self.assertEqual(20, toy2.numNodes())
        self.assertEqual(56, toy2.numPorts())
        self.assertEqual(52, toy2.numLinks())
        self.assertEqual(P10, toy2.findPeerPortOfPort(P9).name)
        self.assertFalse(toy2.findPeerPortOfPort(P9).dcn_facing)
        self.assertEqual(P12, toy2.findPeerPortOfPort(P11).name)
        self.assertTrue(toy2.findPeerPortOfPort(P11).dcn_facing)
        self.assertEqual(-1, toy2.findCapacityOfPath('non-existent-path'))
        self.assertEqual(toy2.findCapacityOfPath(PATH1),
                         toy2.findCapacityOfPath(PATH2))
        self.assertEqual(200000, toy2.findCapacityOfPath(PATH1))
        # verify IP prefix assignment
        ip_aggregate_1 = ipaddress.ip_network('172.16.0.0/24')
        ip_aggregate_2 = ipaddress.ip_network('172.16.1.0/24')
        ip_prefix1 = toy2.findHostPrefixOfToR(TOR1)
        self.assertTrue(ip_prefix1.subnet_of(ip_aggregate_1))
        self.assertFalse(ip_prefix1.subnet_of(ip_aggregate_2))
        ip_prefix2 = toy2.findHostPrefixOfToR(TOR2)
        self.assertTrue(ip_prefix2.subnet_of(ip_aggregate_2))
        self.assertFalse(ip_prefix2.subnet_of(ip_aggregate_1))
        # Verify topology query results.
        path_set = toy2.findPathSetOfAggrBlockPair(C1AB1, C3AB1)
        expected_path_set = {
            (C1AB1, C3AB1): [(C1AB1, C3AB1)],
            (C1AB1, C2AB1, C3AB1): [(C1AB1, C2AB1), (C2AB1, C3AB1)]
        }
        self.assertEqual(expected_path_set, path_set)
        expected_filtered_set = {
            (C1AB1, C2AB1, C3AB1): [(C1AB1, C2AB1), (C2AB1, C3AB1)]
        }
        filtered_path_set = filterPathSetWithSeg(path_set, (C2AB1, C3AB1))
        self.assertEqual(expected_filtered_set, filtered_path_set)
        # Verify findLinksOfPath()
        self.assertEqual(None, toy2.findLinksOfPath('non-existent-path'))
        links = toy2.findLinksOfPath(PATH1)
        self.assertEqual(2, len(links))
        self.assertEqual([LINK1, LINK2], [link.name for link in links])

    def test_toy2_traffic_demand(self):
        toy2_traffic = loadTraffic(TOY2_TRAFFIC_PATH)
        self.assertNotEqual(None, toy2_traffic)
        self.assertEqual(traffic_pb2.TrafficDemand.DemandType.LEVEL_AGGR_BLOCK,
                         toy2_traffic.type)
        self.assertEqual(2, len(toy2_traffic.demands))
        self.assertEqual('toy2-c1-ab1', toy2_traffic.demands[0].src)
        self.assertEqual('toy2-c3-ab1', toy2_traffic.demands[0].dst)
        self.assertEqual(300000, toy2_traffic.demands[0].volume_mbps)
        self.assertEqual('toy2-c3-ab1', toy2_traffic.demands[1].src)
        self.assertEqual('toy2-c1-ab1', toy2_traffic.demands[1].dst)
        self.assertEqual(100000, toy2_traffic.demands[1].volume_mbps)

    def test_toy2_traffic_construction(self):
        toy2_traffic = Traffic(TOY2_TRAFFIC_PATH)
        self.assertEqual(2, len(toy2_traffic.getAllDemands()))
        self.assertEqual({('toy2-c1-ab1', 'toy2-c3-ab1'): 300000,
                          ('toy2-c3-ab1', 'toy2-c1-ab1'): 100000},
                         toy2_traffic.getAllDemands())

if __name__ == "__main__":
    unittest.main()
