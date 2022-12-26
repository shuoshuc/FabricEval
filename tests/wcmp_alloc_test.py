import ipaddress
import unittest

from localTE.group_reduction import GroupReduction
from localTE.wcmp_alloc import WCMPAllocation, loadTESolution
from topology.topogen import generateToy3
from topology.topology import Topology, loadTopo

TOY2_TOPO_PATH = 'tests/data/toy2.textproto'
TOY2_SOL_PATH = 'tests/data/te_sol_toy2.textproto'
C1AB1 = 'toy2-c1-ab1'
C3AB1 = 'toy2-c3-ab1'
# Toy3
TOY3_SOL_PATH = 'tests/data/toy3_te_sol.textproto'
TOY3_C1 = 'toy3-c1-ab1'
TOY3_LINK1 = 'toy3-c64-ab1-s3i1-p29:toy3-c15-ab1-s3i1-p125'
TOY3_LINK2 = 'toy3-c42-ab1-s3i4-p105:toy3-c54-ab1-s3i4-p83'
TOY3_NODE1 = 'toy3-c1-ab1-s3i1'

class TestWCMPAlloc(unittest.TestCase):
    def test_load_invalid_te_solution(self):
        self.assertEqual(None, loadTESolution(''))

    def test_load_valid_te_solution(self):
        sol = loadTESolution(TOY2_SOL_PATH)
        self.assertNotEqual(None, sol)

    def test_toy2_sol_entries(self):
        sol = loadTESolution(TOY2_SOL_PATH)
        # expects 2 TEIntents
        self.assertEqual(2, len(sol.te_intents))
        aggr_block_set = set()
        for te_intent in sol.te_intents:
            aggr_block_set.add(te_intent.target_block)
        self.assertEqual(set({C1AB1, C3AB1}), aggr_block_set)
        # expects 2 prefixes for c1-ab1, and 1 prefix for c3-ab1.
        ip_aggregate = ipaddress.ip_network('172.16.0.0/16')
        for te_intent in sol.te_intents:
            if te_intent.target_block == C1AB1:
                self.assertEqual(2, len(te_intent.prefix_intents))
            if te_intent.target_block == C3AB1:
                self.assertEqual(1, len(te_intent.prefix_intents))
            # verifies that each dst_prefix is within the cluster aggregate.
            for prefix_intent in te_intent.prefix_intents:
                ipa = ipaddress.ip_network(prefix_intent.dst_prefix + '/' +
                                           str(prefix_intent.mask))
                self.assertTrue(ipa.subnet_of(ip_aggregate))
                # nexthop weight should be positive.
                for nexthop in prefix_intent.nexthop_entries:
                    self.assertGreater(nexthop.weight, 0.0)

    def test_toy3_intent_distribution(self):
        toy3 = Topology('', input_proto=generateToy3())
        wcmp_alloc = WCMPAllocation(toy3, TOY3_SOL_PATH)
        self.assertEqual(65, len(wcmp_alloc._worker_map.keys()))
        c1_worker = wcmp_alloc._worker_map[TOY3_C1]
        self.assertEqual(TOY3_C1, c1_worker._target_block)
        self.assertEqual(TOY3_C1, c1_worker._te_intent.target_block)

    def test_toy3_generated_groups(self):
        toy3 = Topology('', input_proto=generateToy3())
        wcmp_alloc = WCMPAllocation(toy3, TOY3_SOL_PATH)
        wcmp_alloc.run()
        c1_worker = wcmp_alloc._worker_map[TOY3_C1]
        # Verify there exist 4 nodes * (SRC and TRANSIT) = 8 sets of groups.
        self.assertEqual(8, len(c1_worker.groups.values()))
        for node, _, _ in c1_worker.groups.keys():
            # Verify node has non-zero ECMP utilization.
            self.assertTrue(toy3.getNodeByName(node).getECMPUtil() > 0)
            self.assertTrue(toy3.getNodeByName(node).getNumGroups() > 0)
        link_util = toy3.dumpRealLinkUtil()
        # Verify real link utilization.
        self.assertTrue(link_util[TOY3_LINK1] > 0.68)
        self.assertTrue(link_util[TOY3_LINK2] > 0.36)
        ecmp_util = toy3.dumpECMPUtil()
        # Verify node ECMP utilization.
        self.assertTrue(ecmp_util[TOY3_NODE1][0] > 0.025)
        self.assertEqual(73, ecmp_util[TOY3_NODE1][1])

class TestGroupReduction(unittest.TestCase):
    def test_single_switch_single_group_1(self):
        group_reduction = GroupReduction([[1, 2, 3, 4]], 16*1024)
        self.assertEqual([[1, 2, 3, 4]], group_reduction.solve_sssg())
        self.assertEqual([[1, 2, 3, 4]], group_reduction.table_fitting_sssg())

    def test_single_switch_single_group_2(self):
        group_reduction = GroupReduction([[20, 40, 60, 80]], 16*1024)
        self.assertEqual([[1, 2, 3, 4]], group_reduction.solve_sssg())
        # EuroSys heuristic does not perform lossless reduction if groups fit.
        self.assertEqual([[20, 40, 60, 80]],
                         group_reduction.table_fitting_sssg())

    def test_single_switch_single_group_3(self):
        group_reduction = GroupReduction([[10.5, 20.1, 31.0, 39.7]], 10)
        self.assertEqual([[1, 2, 3, 4]], group_reduction.solve_sssg())
        self.assertEqual([[1, 2, 3, 4]], group_reduction.table_fitting_sssg())

    def test_single_switch_single_group_4(self):
        group_reduction = GroupReduction([[i + 0.1 for i in range(1, 17)]],
                                         16*1024)
        self.assertEqual([[(i + 0.1) * 10 for i in range(1, 17)]],
                         group_reduction.solve_sssg())
        self.assertEqual([list(range(1, 17))],
                         group_reduction.table_fitting_sssg())

    def test_single_switch_single_group_5(self):
        group_reduction = GroupReduction([[2000.01, 0, 0, 0]], 10)
        self.assertEqual([[1, 0, 0, 0]], group_reduction.solve_sssg())
        self.assertEqual([[1, 0, 0, 0]], group_reduction.table_fitting_sssg())

    def test_single_switch_multi_group_1(self):
        group_reduction = GroupReduction([[1, 2], [3, 4]], 16*1024)
        self.assertEqual([[1, 2], [3, 4]], group_reduction.solve_ssmg())
        self.assertEqual([[1, 2], [3, 4]], group_reduction.table_fitting_ssmg())

    def test_single_switch_multi_group_2(self):
        group_reduction = GroupReduction([[1.1, 2.1], [3.1, 4.1]], 16*1024)
        self.assertEqual([[1, 2], [3, 4]], group_reduction.table_fitting_ssmg())

    def test_single_switch_multi_group_3(self):
        group_reduction = GroupReduction([[1, 0, 0], [0, 2, 4]], 5)
        # Verify that zeroes are correctly stripped and unstripped.
        self.assertEqual([[1, 0, 0], [0, 1, 2]],
                         group_reduction.solve_ssmg())
        self.assertEqual([[1, 0, 0], [0, 1, 2]],
                         group_reduction.table_fitting_ssmg())

if __name__ == "__main__":
    unittest.main()
