import ipaddress
import unittest

import numpy as np
import proto.traffic_pb2 as traffic_pb2

import common.flags as FLAG
from topology.topogen import (generateF1, generateF2, generateToy3,
                              generateToy4, generateToy5)
from topology.topology import Topology, filterPathSetWithSeg, loadTopo
from traffic.tmgen import tmgen
from traffic.traffic import Traffic, loadTraffic

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
TOY2_PATH = 'tests/data/toy2_topo.textproto'
TOY2_TRAFFIC_PATH = 'tests/data/toy2_traffic.textproto'
TOR1 = 'toy2-c1-ab1-s1i1'
TOR2 = 'toy2-c3-ab1-s1i4'
# Toy3 entities.
TOY3_PATH1 = 'toy3-c1-ab1:toy3-c2-ab1'
TOY3_PATH2 = 'toy3-c1-ab1:toy3-c65-ab1'
TOY3_PATH3 = 'toy3-c64-ab1:toy3-c65-ab1'
TOY3_LINK1 = 'toy3-c1-ab1-s3i1-p1:toy3-c2-ab1-s3i1-p1'
TOY3_LINK2 = 'toy3-c1-ab1-s3i4-p1:toy3-c2-ab1-s3i4-p1'
TOY3_PORT1 = 'toy3-c65-ab1-s3i2-p1'
TOY3_PEER_PORT1 = 'toy3-c1-ab1-s3i2-p127'
TOY3_PORT2 = 'toy3-c1-ab1-s3i1-p2'
TOY3_PEER_PORT2 = 'toy3-c1-ab1-s2i1-p1'
TOY3_PORT3 = 'toy3-c2-ab1-s2i1-p2'
TOY3_PEER_PORT3 = 'toy3-c2-ab1-s1i1-p1'
TOY3_AGGR_BLOCK1 = 'toy3-c65-ab1'
TOY3_AGGR_BLOCK2 = 'toy3-c1-ab1'
TOY3_TOR1 = 'toy3-c65-ab1-s1i1'
TOY3_TOR2 = 'toy3-c1-ab1-s1i1'
# Toy4 entities.
TOY4_PATH1 = 'toy4-c1-ab1:toy4-c2-ab1'
TOY4_LINK1 = 'toy4-c1-ab1-s3i1-p1:toy4-c2-ab1-s3i1-p1'
TOY4_PORT1 = 'toy4-c1-ab1-s3i1-p1'
TOY4_PEER_PORT1 = 'toy4-c2-ab1-s3i1-p1'
TOY4_PORT2 = 'toy4-c1-ab1-s3i1-p2'
TOY4_PEER_PORT2 = 'toy4-c1-ab1-s2i1-p1'
TOY4_PORT3 = 'toy4-c1-ab1-s2i1-p2'
TOY4_PEER_PORT3 = 'toy4-c1-ab1-s1i1-p1'
TOY4_AGGR_BLOCK1 = 'toy4-c1-ab1'
TOY4_AGGR_BLOCK2 = 'toy4-c5-ab1'
TOY4_TOR1 = 'toy4-c1-ab1-s1i1'
# Toy5 entities.
TOY5_PATH1 = 'toy5-c1-ab1:toy5-c2-ab1'
TOY5_PATH2 = 'toy5-c1-ab1:toy5-c33-ab1'
TOY5_PATH3 = 'toy5-c32-ab1:toy5-c33-ab1'
TOY5_LINK1 = 'toy5-c1-ab1-s3i1-p1:toy5-c2-ab1-s3i1-p1'
TOY5_PORT1 = 'toy5-c1-ab1-s3i1-p1'
TOY5_PEER_PORT1 = 'toy5-c2-ab1-s3i1-p1'
TOY5_PORT2 = 'toy5-c1-ab1-s3i1-p2'
TOY5_PEER_PORT2 = 'toy5-c1-ab1-s2i1-p1'
TOY5_PORT3 = 'toy5-c1-ab1-s2i1-p2'
TOY5_PEER_PORT3 = 'toy5-c1-ab1-s1i1-p1'
TOY5_AGGR_BLOCK1 = 'toy5-c1-ab1'
TOY5_AGGR_BLOCK2 = 'toy5-c2-ab1'
TOY5_TOR1 = 'toy5-c1-ab1-s1i1'
# F1 entities.
F1_PATH1 = 'f1-c1-ab1:f1-c2-ab1'
F1_PATH2 = 'f1-c1-ab1:f1-c33-ab1'
F1_PATH3 = 'f1-c32-ab1:f1-c33-ab1'
F1_LINK1 = 'f1-c1-ab1-s3i1-p1:f1-c2-ab1-s3i1-p1'
F1_PORT1 = 'f1-c1-ab1-s3i1-p1'
F1_PEER_PORT1 = 'f1-c2-ab1-s3i1-p1'
F1_PORT2 = 'f1-c1-ab1-s3i1-p2'
F1_PEER_PORT2 = 'f1-c1-ab1-s2i1-p1'
F1_PORT3 = 'f1-c1-ab1-s2i1-p2'
F1_PEER_PORT3 = 'f1-c1-ab1-s1i1-p1'
F1_AGGR_BLOCK1 = 'f1-c1-ab1'
F1_AGGR_BLOCK2 = 'f1-c2-ab1'
F1_TOR1 = 'f1-c1-ab1-s1i1'
# F2 entities.
F2_PATH1 = 'f2-c1-ab1:f2-c2-ab1'
F2_PATH2 = 'f2-c1-ab1:f2-c5-ab1'
F2_PATH3 = 'f2-c4-ab1:f2-c5-ab1'
F2_LINK1 = 'f2-c1-ab1-s3i1-p1:f2-c2-ab1-s3i1-p1'
F2_PORT1 = 'f2-c1-ab1-s3i1-p1'
F2_PEER_PORT1 = 'f2-c2-ab1-s3i1-p1'
F2_PORT2 = 'f2-c1-ab1-s3i1-p2'
F2_PEER_PORT2 = 'f2-c1-ab1-s2i1-p1'
F2_PORT3 = 'f2-c1-ab1-s2i1-p2'
F2_PEER_PORT3 = 'f2-c1-ab1-s1i1-p1'
F2_AGGR_BLOCK1 = 'f2-c1-ab1'
F2_AGGR_BLOCK2 = 'f2-c2-ab1'
F2_TOR1 = 'f2-c1-ab1-s1i1'


class TestLoadToyNet(unittest.TestCase):
    def test_load_invalid_topo(self):
        self.assertEqual(None, loadTopo(''))

    def test_load_invalid_traffic(self):
        self.assertEqual(None, loadTraffic(''))

    def test_load_valid_toynet(self):
        toy1 = loadTopo(TOY2_PATH)
        self.assertNotEqual(None, toy1)

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
        toy2 = Topology(TOY2_PATH)
        toy2_traffic = Traffic(toy2, TOY2_TRAFFIC_PATH)
        self.assertEqual(2, len(toy2_traffic.getAllDemands()))
        self.assertEqual(
            {
                ('toy2-c1-ab1', 'toy2-c3-ab1'): 300000,
                ('toy2-c3-ab1', 'toy2-c1-ab1'): 100000
            }, toy2_traffic.getAllDemands())

    def test_toy2_topology_serialization(self):
        toy2 = Topology(TOY2_PATH)
        toy2_proto = toy2.serialize()
        # check network
        self.assertEqual('toy2', toy2_proto.name)
        self.assertEqual(3, len(toy2_proto.clusters))
        self.assertEqual(6, len(toy2_proto.paths))


class TestLoadToy3Net(unittest.TestCase):
    def test_toy3_topology_construction(self):
        FLAG.P_LINK_FAILURE = 0.0
        toy3 = Topology('', input_proto=generateToy3())
        self.assertEqual(65, toy3.numClusters())
        # 8 + 32 nodes per cluster
        self.assertEqual(65 * 40, toy3.numNodes())
        # 8 * 32 * 2 * 65 S1-S2 links per cluster, 64 * 4 * 2 * 65 S2-S3 links,
        # 64 * 4 * 65 S3-S3 links.
        self.assertEqual(8 * 32 * 2 * 65 + 64 * 4 * 2 * 65 + 64 * 4 * 65,
                         toy3.numLinks())
        self.assertEqual(65 * 64, len(toy3.getAllPaths()))
        # Path between two 40G clusters: 4 * 40
        self.assertEqual(160000, toy3.findCapacityOfPath(TOY3_PATH1))
        # Path between a 40G cluster and a 200G cluster: 4 * 40
        self.assertEqual(160000, toy3.findCapacityOfPath(TOY3_PATH2))
        # Path between two 200G clusters: 4 * 200
        self.assertEqual(160000, toy3.findCapacityOfPath(TOY3_PATH1))
        self.assertEqual(800000, toy3.findCapacityOfPath(TOY3_PATH3))
        links = [l.name for l in toy3.findLinksOfPath(TOY3_PATH1)]
        self.assertTrue(TOY3_LINK1 in links)
        self.assertTrue(TOY3_LINK2 in links)
        # Verify S3-S3 port and peer.
        self.assertEqual(TOY3_PEER_PORT1,
                         toy3.findPeerPortOfPort(TOY3_PORT1).name)
        # Verify that all DCN ports have odd port indices.
        p1 = toy3.getPortByName(TOY3_PORT1)
        self.assertTrue(p1.dcn_facing)
        self.assertEqual(1, p1.index % 2)
        pp1 = toy3.getPortByName(TOY3_PEER_PORT1)
        self.assertTrue(pp1.dcn_facing)
        self.assertEqual(1, pp1.index % 2)
        # Verify S2-S3 port and peer.
        self.assertEqual(TOY3_PEER_PORT2,
                         toy3.findPeerPortOfPort(TOY3_PORT2).name)
        # Verify that S2-facing S3 ports have even indices.
        p2 = toy3.getPortByName(TOY3_PORT2)
        self.assertFalse(p2.dcn_facing)
        self.assertEqual(0, p2.index % 2)
        # Verify that S3-facing S2 ports have odd indices.
        pp2 = toy3.getPortByName(TOY3_PEER_PORT2)
        self.assertFalse(pp2.dcn_facing)
        self.assertEqual(1, pp2.index % 2)
        # Verify S1-S2 port and peer.
        self.assertEqual(TOY3_PEER_PORT3,
                         toy3.findPeerPortOfPort(TOY3_PORT3).name)
        # Verify that S1-facing S2 ports have even indices.
        p3 = toy3.getPortByName(TOY3_PORT3)
        self.assertFalse(p3.dcn_facing)
        self.assertEqual(0, p3.index % 2)
        # Verify that S2-facing S1 ports have odd indices.
        pp3 = toy3.getPortByName(TOY3_PEER_PORT3)
        self.assertFalse(pp3.dcn_facing)
        self.assertEqual(1, pp3.index % 2)
        # Verify port and AggrBlock has correct child-parent relationship.
        self.assertEqual(TOY3_AGGR_BLOCK1,
                         toy3.findAggrBlockOfPort(TOY3_PORT1).name)
        self.assertTrue(toy3.hasAggrBlock(TOY3_AGGR_BLOCK1))
        # Verify the 'virutal' parent of ToRs.
        self.assertEqual(TOY3_AGGR_BLOCK1,
                         toy3.findAggrBlockOfToR(TOY3_TOR1).name)
        self.assertEqual(TOY3_AGGR_BLOCK2,
                         toy3.findAggrBlockOfToR(TOY3_TOR2).name)
        # Verify the stage and index of ToR1.
        self.assertEqual(1, toy3.getNodeByName(TOY3_TOR1).stage)
        self.assertEqual(1, toy3.getNodeByName(TOY3_TOR1).index)

    def test_toy3_traffic_construction1(self):
        toy3 = Topology('', input_proto=generateToy3())
        traffic_proto = tmgen(tor_level=False,
                              cluster_vector=np.array([1] * 22 + [2.5] * 22 +
                                                      [5] * 21),
                              num_nodes=32,
                              model='flat',
                              dist='',
                              netname='toy3')
        toy3_traffic = Traffic(toy3, '', traffic_proto)
        self.assertEqual(traffic_pb2.TrafficDemand.DemandType.LEVEL_AGGR_BLOCK,
                         toy3_traffic.getDemandType())
        self.assertEqual(64 * 65, len(toy3_traffic.getAllDemands()))
        # Flat demand matrix has the same volume in both directions.
        self.assertEqual(
            80000, toy3_traffic.getDemand(TOY3_AGGR_BLOCK1, TOY3_AGGR_BLOCK2))
        self.assertEqual(
            80000, toy3_traffic.getDemand(TOY3_AGGR_BLOCK2, TOY3_AGGR_BLOCK1))

    def test_toy3_traffic_construction2(self):
        FLAG.P_SPARSE = 0.1
        toy3 = Topology('', input_proto=generateToy3())
        traffic_proto = tmgen(tor_level=False,
                              cluster_vector=np.array([1] * 22 + [2.5] * 22 +
                                                      [5] * 21),
                              num_nodes=32,
                              model='gravity',
                              dist='exp',
                              netname='toy3')
        toy3_traffic = Traffic(toy3, '', traffic_proto)
        self.assertEqual(traffic_pb2.TrafficDemand.DemandType.LEVEL_AGGR_BLOCK,
                         toy3_traffic.getDemandType())
        self.assertTrue(64 * 65 >= len(toy3_traffic.getAllDemands()))
        # Bernoulli distribution may not generate the exact same number as
        # requested, so conservatively under-estimates by 20%.
        non_empty_blocks = round((1 - FLAG.P_SPARSE) * 65 * 0.8)
        num_demands = (non_empty_blocks - 1) * non_empty_blocks
        self.assertTrue(num_demands <= len(toy3_traffic.getAllDemands()))

    def test_toy3_traffic_construction3(self):
        FLAG.P_SPARSE = 0.0
        toy3 = Topology('', input_proto=generateToy3())
        traffic_proto = tmgen(tor_level=False,
                              cluster_vector=np.array([1] * 22 + [2.5] * 22 +
                                                      [5] * 21),
                              num_nodes=32,
                              model='gravity',
                              dist='exp',
                              netname='toy3')
        toy3_traffic = Traffic(toy3, '', traffic_proto)
        self.assertEqual(traffic_pb2.TrafficDemand.DemandType.LEVEL_AGGR_BLOCK,
                         toy3_traffic.getDemandType())
        self.assertEqual(64 * 65, len(toy3_traffic.getAllDemands()))
        # Verify the sum of all demands from AggrBlock2 does not exceed its
        # total capacity.
        tot_traffic = 0
        for i in range(2, 66):
            dst_block_name = f'toy3-c{i}-ab1'
            tot_traffic += toy3_traffic.getDemand(TOY3_AGGR_BLOCK2,
                                                  dst_block_name)
        self.assertTrue(tot_traffic < 40000 * 256)
        # Verify the sum of all demands from AggrBlock1 does not exceed its
        # total capacity. The effective total capacity is not 200G * 256, but
        # a function of the peer block speed due to speed auto-negotiation.
        tot_traffic = 0
        for i in range(1, 65):
            dst_block_name = f'toy3-c{i}-ab1'
            tot_traffic += toy3_traffic.getDemand(TOY3_AGGR_BLOCK1,
                                                  dst_block_name)
        self.assertTrue(
            tot_traffic < 40000 * 22 * 4 + 100000 * 22 * 4 + 200000 * 20 * 4)

    def test_toy3_traffic_construction4(self):
        toy3 = Topology('', input_proto=generateToy3())
        traffic_proto = tmgen(tor_level=True,
                              cluster_vector=np.array([1] * 22 + [2.5] * 22 +
                                                      [5] * 21),
                              num_nodes=32,
                              model='flat',
                              dist='',
                              netname='toy3')
        toy3_traffic = Traffic(toy3, '', traffic_proto)
        self.assertEqual(traffic_pb2.TrafficDemand.DemandType.LEVEL_TOR,
                         toy3_traffic.getDemandType())
        self.assertEqual(64 * 65, len(toy3_traffic.getAllDemands()))
        # Flat demand matrix has the same volume in both directions.
        # The inter-block demand should be 153 Mbps (per-ToR) * 32 peer ToRs *
        # 32 sister ToRs.
        self.assertEqual(
            115 * 32 * 32,
            toy3_traffic.getDemand(TOY3_AGGR_BLOCK1, TOY3_AGGR_BLOCK2))
        self.assertEqual(
            115 * 32 * 32,
            toy3_traffic.getDemand(TOY3_AGGR_BLOCK2, TOY3_AGGR_BLOCK1))


class TestLoadToy4Net(unittest.TestCase):
    def test_toy4_topology_construction(self):
        toy4 = Topology('', input_proto=generateToy4())
        self.assertEqual(5, toy4.numClusters())
        # 8 + 32 nodes per cluster
        self.assertEqual(5 * 12, toy4.numNodes())
        self.assertEqual(5 * 4, len(toy4.getAllPaths()))
        # All paths in Toy4 have 160G capacity.
        for path in toy4.getAllPaths().values():
            self.assertEqual(160000, path.capacity)
        links = [l.name for l in toy4.findLinksOfPath(TOY4_PATH1)]
        self.assertTrue(TOY4_LINK1 in links)
        # Verify S3-S3 port and peer.
        self.assertEqual(TOY4_PEER_PORT1,
                         toy4.findPeerPortOfPort(TOY4_PORT1).name)
        # Verify that all DCN ports have odd port indices.
        p1 = toy4.getPortByName(TOY4_PORT1)
        self.assertEqual(TOY4_PORT1, p1.name)
        self.assertTrue(p1.dcn_facing)
        self.assertEqual(1, p1.index % 2)
        pp1 = toy4.getPortByName(TOY4_PEER_PORT1)
        self.assertTrue(pp1.dcn_facing)
        self.assertEqual(1, pp1.index % 2)
        # Verify S2-S3 port and peer.
        self.assertEqual(TOY4_PEER_PORT2,
                         toy4.findPeerPortOfPort(TOY4_PORT2).name)
        # Verify that S2-facing S3 ports have even indices.
        p2 = toy4.getPortByName(TOY4_PORT2)
        self.assertFalse(p2.dcn_facing)
        self.assertEqual(0, p2.index % 2)
        # Verify that S3-facing S2 ports have odd indices.
        pp2 = toy4.getPortByName(TOY4_PEER_PORT2)
        self.assertFalse(pp2.dcn_facing)
        self.assertEqual(1, pp2.index % 2)
        # Verify S1-S2 port and peer.
        self.assertEqual(TOY4_PEER_PORT3,
                         toy4.findPeerPortOfPort(TOY4_PORT3).name)
        # Verify that S1-facing S2 ports have even indices.
        p3 = toy4.getPortByName(TOY4_PORT3)
        self.assertFalse(p3.dcn_facing)
        self.assertEqual(0, p3.index % 2)
        # Verify that S2-facing S1 ports have odd indices.
        pp3 = toy4.getPortByName(TOY4_PEER_PORT3)
        self.assertFalse(pp3.dcn_facing)
        self.assertEqual(1, pp3.index % 2)
        # Verify the 'virutal' parent of ToRs.
        self.assertEqual(TOY4_AGGR_BLOCK1,
                         toy4.findAggrBlockOfToR(TOY4_TOR1).name)
        # Verify the stage and index of ToR1.
        self.assertEqual(1, toy4.getNodeByName(TOY4_TOR1).stage)

    def test_toy4_traffic_construction(self):
        toy4 = Topology('', input_proto=generateToy4())
        traffic_proto = tmgen(tor_level=False,
                              cluster_vector=np.array([1] * 5),
                              num_nodes=4,
                              model='single',
                              dist='',
                              netname='toy4')
        toy4_traffic = Traffic(toy4, '', traffic_proto)
        self.assertEqual(traffic_pb2.TrafficDemand.DemandType.LEVEL_AGGR_BLOCK,
                         toy4_traffic.getDemandType())
        # There should be only 1 demand.
        self.assertEqual(1, len(toy4_traffic.getAllDemands()))
        self.assertTrue((TOY4_AGGR_BLOCK1, TOY4_AGGR_BLOCK2) in \
                        toy4_traffic.getAllDemands())
        self.assertEqual(
            40000 * 4 * 0.5,
            toy4_traffic.getAllDemands()[(TOY4_AGGR_BLOCK1, TOY4_AGGR_BLOCK2)])


class TestLoadToy5Net(unittest.TestCase):
    def test_toy5_topology_construction(self):
        toy5 = Topology('', input_proto=generateToy5())
        self.assertEqual(33, toy5.numClusters())
        # 8 + 16 nodes per cluster
        self.assertEqual(33 * 24, toy5.numNodes())
        self.assertEqual(33 * 32, len(toy5.getAllPaths()))
        # Path between two 40G clusters: 4 * 40
        self.assertEqual(160000, toy5.findCapacityOfPath(TOY5_PATH1))
        # Path between a 40G cluster and a 200G cluster: 4 * 40
        self.assertEqual(160000, toy5.findCapacityOfPath(TOY5_PATH2))
        # Path between two 200G clusters: 4 * 200
        self.assertEqual(800000, toy5.findCapacityOfPath(TOY5_PATH3))
        links = [l.name for l in toy5.findLinksOfPath(TOY5_PATH1)]
        self.assertTrue(TOY5_LINK1 in links)
        # Verify S3-S3 port and peer.
        self.assertEqual(TOY5_PEER_PORT1,
                         toy5.findPeerPortOfPort(TOY5_PORT1).name)
        # Verify that DCN ports have odd port indices.
        p1 = toy5.getPortByName(TOY5_PORT1)
        self.assertEqual(TOY5_PORT1, p1.name)
        self.assertTrue(p1.dcn_facing)
        self.assertEqual(1, p1.index % 2)
        pp1 = toy5.getPortByName(TOY5_PEER_PORT1)
        self.assertTrue(pp1.dcn_facing)
        self.assertEqual(1, pp1.index % 2)
        # Verify S2-S3 port and peer.
        self.assertEqual(TOY5_PEER_PORT2,
                         toy5.findPeerPortOfPort(TOY5_PORT2).name)
        # Verify that S2-facing S3 ports have even indices.
        p2 = toy5.getPortByName(TOY5_PORT2)
        self.assertFalse(p2.dcn_facing)
        self.assertEqual(0, p2.index % 2)
        # Verify that S3-facing S2 ports have odd indices.
        pp2 = toy5.getPortByName(TOY5_PEER_PORT2)
        self.assertFalse(pp2.dcn_facing)
        self.assertEqual(1, pp2.index % 2)
        # Verify S1-S2 port and peer.
        self.assertEqual(TOY5_PEER_PORT3,
                         toy5.findPeerPortOfPort(TOY5_PORT3).name)
        # Verify that S1-facing S2 ports have even indices.
        p3 = toy5.getPortByName(TOY5_PORT3)
        self.assertFalse(p3.dcn_facing)
        self.assertEqual(0, p3.index % 2)
        # Verify that S2-facing S1 ports have odd indices.
        pp3 = toy5.getPortByName(TOY5_PEER_PORT3)
        self.assertFalse(pp3.dcn_facing)
        self.assertEqual(1, pp3.index % 2)
        # Verify the 'virutal' parent of ToRs.
        self.assertEqual(TOY5_AGGR_BLOCK1,
                         toy5.findAggrBlockOfToR(TOY5_TOR1).name)
        # Verify the stage and index of ToR1.
        self.assertEqual(1, toy5.getNodeByName(TOY5_TOR1).stage)

    def test_toy5_traffic_construction1(self):
        toy5 = Topology('', input_proto=generateToy5())
        traffic_proto = tmgen(tor_level=False,
                              cluster_vector=np.array([1] * 11 + [2.5] * 11 +
                                                      [5] * 11),
                              num_nodes=16,
                              model='flat',
                              dist='',
                              netname='toy5')
        toy5_traffic = Traffic(toy5, '', traffic_proto)
        self.assertEqual(traffic_pb2.TrafficDemand.DemandType.LEVEL_AGGR_BLOCK,
                         toy5_traffic.getDemandType())
        self.assertEqual(33 * 32, len(toy5_traffic.getAllDemands()))
        # Flat demand matrix has the same volume in both directions.
        # The inter-block demand should be 160000 Mbps when tor_level=False.
        self.assertEqual(
            160000, toy5_traffic.getDemand(TOY5_AGGR_BLOCK1, TOY5_AGGR_BLOCK2))
        self.assertEqual(
            160000, toy5_traffic.getDemand(TOY5_AGGR_BLOCK2, TOY5_AGGR_BLOCK1))

    def test_toy5_traffic_construction2(self):
        toy5 = Topology('', input_proto=generateToy5())
        traffic_proto = tmgen(tor_level=False,
                              cluster_vector=np.array([1] * 11 + [2.5] * 11 +
                                                      [5] * 11),
                              num_nodes=16,
                              model='gravity',
                              dist='exp',
                              netname='toy5')
        toy5_traffic = Traffic(toy5, '', traffic_proto)
        self.assertEqual(traffic_pb2.TrafficDemand.DemandType.LEVEL_AGGR_BLOCK,
                         toy5_traffic.getDemandType())
        self.assertEqual(33 * 32, len(toy5_traffic.getAllDemands()))
        # Gravity demand matrix should not set the 40G blocks to empty.
        self.assertNotEqual(
            0, toy5_traffic.getDemand(TOY5_AGGR_BLOCK1, TOY5_AGGR_BLOCK2))
        self.assertNotEqual(
            0, toy5_traffic.getDemand(TOY5_AGGR_BLOCK2, TOY5_AGGR_BLOCK1))


class TestLoadF1Net(unittest.TestCase):
    def test_f1_topology_construction(self):
        f1 = Topology('', input_proto=generateF1())
        self.assertEqual(33, f1.numClusters())
        # 8 + 16 nodes per cluster
        self.assertEqual(33 * 24, f1.numNodes())
        self.assertEqual(33 * 32, len(f1.getAllPaths()))
        # Path between two 40G clusters: 4 * 40
        self.assertEqual(160000, f1.findCapacityOfPath(F1_PATH1))
        # Path between a 40G cluster and a 200G cluster: 4 * 40
        self.assertEqual(160000, f1.findCapacityOfPath(F1_PATH2))
        # Path between two 200G clusters: 4 * 200
        self.assertEqual(800000, f1.findCapacityOfPath(F1_PATH3))
        links = [l.name for l in f1.findLinksOfPath(F1_PATH1)]
        self.assertTrue(F1_LINK1 in links)
        # Verify S3-S3 port and peer.
        self.assertEqual(F1_PEER_PORT1, f1.findPeerPortOfPort(F1_PORT1).name)
        # Verify that DCN ports have odd port indices.
        p1 = f1.getPortByName(F1_PORT1)
        self.assertEqual(F1_PORT1, p1.name)
        self.assertTrue(p1.dcn_facing)
        self.assertEqual(1, p1.index % 2)
        pp1 = f1.getPortByName(F1_PEER_PORT1)
        self.assertTrue(pp1.dcn_facing)
        self.assertEqual(1, pp1.index % 2)
        # Verify S2-S3 port and peer.
        self.assertEqual(F1_PEER_PORT2, f1.findPeerPortOfPort(F1_PORT2).name)
        # Verify that S2-facing S3 ports have even indices.
        p2 = f1.getPortByName(F1_PORT2)
        self.assertFalse(p2.dcn_facing)
        self.assertEqual(0, p2.index % 2)
        # Verify that S3-facing S2 ports have odd indices.
        pp2 = f1.getPortByName(F1_PEER_PORT2)
        self.assertFalse(pp2.dcn_facing)
        self.assertEqual(1, pp2.index % 2)
        # Verify S1-S2 port and peer.
        self.assertEqual(F1_PEER_PORT3, f1.findPeerPortOfPort(F1_PORT3).name)
        # Verify that S1-facing S2 ports have even indices.
        p3 = f1.getPortByName(F1_PORT3)
        self.assertFalse(p3.dcn_facing)
        self.assertEqual(0, p3.index % 2)
        # Verify that S2-facing S1 ports have odd indices.
        pp3 = f1.getPortByName(F1_PEER_PORT3)
        self.assertFalse(pp3.dcn_facing)
        self.assertEqual(1, pp3.index % 2)
        # Verify the 'virutal' parent of ToRs.
        self.assertEqual(F1_AGGR_BLOCK1, f1.findAggrBlockOfToR(F1_TOR1).name)
        # Verify the stage and index of ToR1.
        self.assertEqual(1, f1.getNodeByName(F1_TOR1).stage)


class TestLoadF2Net(unittest.TestCase):
    def test_f2_topology_construction(self):
        f2 = Topology('', input_proto=generateF2())
        self.assertEqual(5, f2.numClusters())
        # 8 + 16 nodes per cluster
        self.assertEqual(5 * 24, f2.numNodes())
        self.assertEqual(5 * 4, len(f2.getAllPaths()))
        # Path between a 40G cluster and a 100G cluster: 32 * 40
        self.assertEqual(1280000, f2.findCapacityOfPath(F2_PATH1))
        # Path between a 40G cluster and a 200G cluster: 32 * 40
        self.assertEqual(1280000, f2.findCapacityOfPath(F2_PATH2))
        # Path between two 200G clusters: 32 * 200
        self.assertEqual(6400000, f2.findCapacityOfPath(F2_PATH3))
        links = [l.name for l in f2.findLinksOfPath(F2_PATH1)]
        self.assertTrue(F2_LINK1 in links)
        # Verify S3-S3 port and peer.
        self.assertEqual(F2_PEER_PORT1, f2.findPeerPortOfPort(F2_PORT1).name)
        # Verify that DCN ports have odd port indices.
        p1 = f2.getPortByName(F2_PORT1)
        self.assertEqual(F2_PORT1, p1.name)
        self.assertTrue(p1.dcn_facing)
        self.assertEqual(1, p1.index % 2)
        pp1 = f2.getPortByName(F2_PEER_PORT1)
        self.assertTrue(pp1.dcn_facing)
        self.assertEqual(1, pp1.index % 2)
        # Verify S2-S3 port and peer.
        self.assertEqual(F2_PEER_PORT2, f2.findPeerPortOfPort(F2_PORT2).name)
        # Verify that S2-facing S3 ports have even indices.
        p2 = f2.getPortByName(F2_PORT2)
        self.assertFalse(p2.dcn_facing)
        self.assertEqual(0, p2.index % 2)
        # Verify that S3-facing S2 ports have odd indices.
        pp2 = f2.getPortByName(F2_PEER_PORT2)
        self.assertFalse(pp2.dcn_facing)
        self.assertEqual(1, pp2.index % 2)
        # Verify S1-S2 port and peer.
        self.assertEqual(F2_PEER_PORT3, f2.findPeerPortOfPort(F2_PORT3).name)
        # Verify that S1-facing S2 ports have even indices.
        p3 = f2.getPortByName(F2_PORT3)
        self.assertFalse(p3.dcn_facing)
        self.assertEqual(0, p3.index % 2)
        # Verify that S2-facing S1 ports have odd indices.
        pp3 = f2.getPortByName(F2_PEER_PORT3)
        self.assertFalse(pp3.dcn_facing)
        self.assertEqual(1, pp3.index % 2)
        # Verify the 'virutal' parent of ToRs.
        self.assertEqual(F2_AGGR_BLOCK1, f2.findAggrBlockOfToR(F2_TOR1).name)
        # Verify the stage and index of ToR1.
        self.assertEqual(1, f2.getNodeByName(F2_TOR1).stage)


if __name__ == "__main__":
    unittest.main()
