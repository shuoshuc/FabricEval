import ipaddress
import unittest

import proto.te_solution_pb2 as te_sol

import common.flags as FLAG
from localTE.group_reduction import GroupReduction
from localTE.wcmp_alloc import WCMPAllocation, loadTESolution
from topology.topogen import generateToy3, generateToy4
from topology.topology import Topology, loadTopo
from traffic.traffic import Traffic

TOY2_TOPO_PATH = 'tests/data/toy2.textproto'
TOY2_SOL_PATH = 'tests/data/te_sol_toy2.textproto'
C1AB1 = 'toy2-c1-ab1'
C3AB1 = 'toy2-c3-ab1'
# Toy3
TOY3_SOL_PATH = 'tests/data/toy3_te_sol.textproto'
TOY3_TM_PATH = 'tests/data/toy3_traffic_gravity.textproto'
TOY3_C1 = 'toy3-c1-ab1'
TOY3_LINK1 = 'toy3-c64-ab1-s3i1-p29:toy3-c15-ab1-s3i1-p125'
TOY3_LINK2 = 'toy3-c42-ab1-s3i4-p105:toy3-c54-ab1-s3i4-p83'
TOY3_NODE1 = 'toy3-c1-ab1-s3i1'
# Toy4
TOY4_TM_PATH = 'tests/data/toy4_traffic.textproto'
TOY4_SOL_PATH = 'tests/data/toy4_te_sol.textproto'
TOY4_C1 = 'toy4-c1-ab1'
TOY4_LINK1 = 'toy4-c4-ab1-s3i1-p1:toy4-c1-ab1-s3i1-p5'
TOY4_NODE1 = 'toy4-c1-ab1-s3i1'

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
        toy3_traffic = Traffic(toy3, TOY3_TM_PATH)
        wcmp_alloc = WCMPAllocation(toy3, toy3_traffic, TOY3_SOL_PATH)
        self.assertEqual(65, len(wcmp_alloc._worker_map.keys()))
        c1_worker = wcmp_alloc._worker_map[TOY3_C1]
        self.assertEqual(TOY3_C1, c1_worker._target_block)
        self.assertEqual(TOY3_C1, c1_worker._te_intent.target_block)

    def test_toy4_generated_groups(self):
        FLAG.GR_ALGO = 'eurosys'
        toy4 = Topology('', input_proto=generateToy4())
        toy4_traffic = Traffic(toy4, TOY4_TM_PATH)
        wcmp_alloc = WCMPAllocation(toy4, toy4_traffic, TOY4_SOL_PATH)
        wcmp_alloc.run()
        c1_worker = wcmp_alloc._worker_map[TOY4_C1]
        # Verify there exist 4 nodes * (SRC and TRANSIT) = 8 sets of groups.
        self.assertEqual(8, len(c1_worker.groups.values()))
        for node, _, _ in c1_worker.groups.keys():
            # Verify node has non-zero ECMP utilization.
            self.assertTrue(toy4.getNodeByName(node).getECMPUtil() > 0)
            self.assertTrue(toy4.getNodeByName(node).getNumGroups() > 0)
        link_util = toy4.dumpRealLinkUtil()
        # Verify real link utilization.
        self.assertTrue(link_util[TOY4_LINK1] > 0.16)
        self.assertTrue(link_util[TOY4_LINK1] < 0.17)
        ecmp_util = toy4.dumpECMPUtil()
        # Verify node ECMP utilization.
        self.assertTrue(ecmp_util[TOY4_NODE1][0] > 0.03)
        self.assertTrue(ecmp_util[TOY4_NODE1][0] < 0.04)
        self.assertEqual(6, ecmp_util[TOY4_NODE1][1])
        demand = toy4.dumpDemandAdmission()
        # Verify node admits all demands.
        self.assertEqual(1.0, demand[TOY4_NODE1][2])

class TestGroupReduction(unittest.TestCase):
    def test_single_switch_single_group_1(self):
        gr = GroupReduction([[1, 2, 3, 4]], te_sol.PrefixIntent.PrefixType.SRC,
                            16*1024)
        self.assertEqual([[1, 2, 3, 4]], gr.sanitize([gr.solve_sssg()]))
        gr.reset()
        self.assertEqual([[1, 2, 3, 4]], gr.table_fitting_sssg())

    def test_single_switch_single_group_2(self):
        gr = GroupReduction([[20, 40, 60, 80]],
                            te_sol.PrefixIntent.PrefixType.SRC,
                            16*1024)
        self.assertEqual([[1, 2, 3, 4]], gr.sanitize([gr.solve_sssg()]))
        gr.reset()
        # EuroSys heuristic does not perform lossless reduction if groups fit.
        self.assertEqual([[20, 40, 60, 80]], gr.table_fitting_sssg())

    def test_single_switch_single_group_3(self):
        gr = GroupReduction([[10.5, 20.1, 31.0, 39.7]],
                            te_sol.PrefixIntent.PrefixType.SRC,
                            10)
        self.assertEqual([[1, 2, 3, 4]], gr.sanitize([gr.solve_sssg()]))
        gr.reset()
        self.assertEqual([[1, 2, 3, 4]], gr.table_fitting_sssg())

    def test_single_switch_single_group_4(self):
        gr = GroupReduction([[i + 0.1 for i in range(1, 17)]],
                            te_sol.PrefixIntent.PrefixType.SRC,
                            16*1024)
        self.assertEqual([[(i + 0.1) * 10 for i in range(1, 17)]],
                         gr.sanitize([gr.solve_sssg()]))
        gr.reset()
        self.assertEqual([list(range(1, 17))], gr.table_fitting_sssg())

    def test_single_switch_single_group_5(self):
        gr = GroupReduction([[2000.01, 0, 0, 0]],
                            te_sol.PrefixIntent.PrefixType.SRC,
                            10)
        self.assertEqual([[1, 0, 0, 0]], gr.sanitize([gr.solve_sssg()]))
        gr.reset()
        self.assertEqual([[1, 0, 0, 0]], gr.table_fitting_sssg())

    def test_single_switch_multi_group_1(self):
        group_reduction = GroupReduction([[1, 2], [3, 4]],
                                         te_sol.PrefixIntent.PrefixType.SRC,
                                         16*1024)
        self.assertEqual([[1, 2], [3, 4]], group_reduction.solve_ssmg())
        group_reduction.reset()
        self.assertEqual([[1, 2], [3, 4]], group_reduction.table_fitting_ssmg())
        group_reduction.reset()
        self.assertEqual([[1, 2], [3, 4]], group_reduction.google_ssmg())

    def test_single_switch_multi_group_2(self):
        group_reduction = GroupReduction([[1.1, 2.1], [3.1, 4.1]],
                                         te_sol.PrefixIntent.PrefixType.SRC,
                                         16*1024)
        self.assertEqual([[1, 2], [3, 4]], group_reduction.table_fitting_ssmg())
        group_reduction.reset()
        self.assertEqual([[1, 2], [3, 4]], group_reduction.google_ssmg())

    def test_single_switch_multi_group_3(self):
        group_reduction = GroupReduction([[1, 0, 0], [0, 2, 4]],
                                         te_sol.PrefixIntent.PrefixType.SRC,
                                         5)
        # Verify that zeroes are correctly stripped and unstripped.
        self.assertEqual([[1, 0, 0], [0, 1, 2]],
                         group_reduction.solve_ssmg())
        group_reduction.reset()
        self.assertEqual([[1, 0, 0], [0, 1, 2]],
                         group_reduction.table_fitting_ssmg())
        group_reduction.reset()
        self.assertEqual([[1, 0, 0], [0, 1, 2]],
                         group_reduction.google_ssmg())

    def test_single_switch_multi_group_4(self):
        group_reduction = GroupReduction([[6, 0, 0], [0, 2, 4], [2, 0, 0]],
                                         te_sol.PrefixIntent.PrefixType.TRANSIT,
                                         5)
        # Google SSMG reduces transit groups to ECMP, does not de-duplicate.
        self.assertEqual([[1, 0, 0], [0, 1, 1], [1, 0, 0]],
                         group_reduction.google_ssmg())

    def test_single_switch_multi_group_5(self):
        FLAG.IMPROVED_HEURISTIC = False
        FLAG.EUROSYS_MOD = False
        group_reduction = GroupReduction([[100, 1, 1], [0, 2, 4]],
                                         te_sol.PrefixIntent.PrefixType.SRC,
                                         4)
        # EuroSys SSMG will reduce to ECMP and give up.
        self.assertEqual([[1, 1, 1], [0, 1, 1]],
                         group_reduction.table_fitting_ssmg())
        group_reduction.reset()
        # Google SSMG prunes the first member of the largest group when simple
        # reduction cannot fit the groups.
        self.assertEqual([[0, 1, 1], [0, 1, 1]],
                         group_reduction.google_ssmg())

    def test_single_switch_multi_group_6(self):
        FLAG.IMPROVED_HEURISTIC = False
        FLAG.EUROSYS_MOD = True
        group_reduction = GroupReduction([[1, 3, 1], [0, 0, 4]],
                                         te_sol.PrefixIntent.PrefixType.SRC,
                                         3)
        # Modified EuroSys SSMG (w/ pruning) will reduce to ECMP and prune the
        # first port of the largest group [1, 1, 1].
        self.assertEqual([[0, 1, 1], [0, 0, 1]],
                         group_reduction.table_fitting_ssmg())

    def test_single_switch_multi_group_7(self):
        FLAG.IMPROVED_HEURISTIC = True
        group_reduction = GroupReduction([[0.01, 0.01, 0.01], [997, 1, 1]],
                                         te_sol.PrefixIntent.PrefixType.SRC,
                                         1000)
        # First group gets 0 entry by simple table carving. But it will be
        # rebalanced to 1 entry, so there is at least one member in the final
        # output.
        groups = group_reduction.table_carving_ssmg()
        self.assertTrue(sum(groups[0]) > 1)
        self.assertTrue(1 in groups[0])

if __name__ == "__main__":
    unittest.main()
