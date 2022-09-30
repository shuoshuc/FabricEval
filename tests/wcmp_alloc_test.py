from topology.topology import loadTopo, Topology
from wcmp_alloc.wcmp_alloc import loadTESolution, WCMPAllocation
from wcmp_alloc.group_reduction import GroupReduction
import unittest
import ipaddress

TOY2_TOPO_PATH = 'tests/data/toy2.textproto'
TOY2_SOL_PATH = 'tests/data/te_sol_toy2.textproto'
C1AB1 = 'toy2-c1-ab1'
C3AB1 = 'toy2-c3-ab1'

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

    def test_wcmp_alloc_intent_distribution(self):
        toy2_topo = Topology(TOY2_TOPO_PATH)
        wcmp_alloc = WCMPAllocation(toy2_topo, TOY2_SOL_PATH)
        wcmp_alloc.run()

class TestGroupReduction(unittest.TestCase):
    def test_single_switch_single_group_1(self):
        group_reduction = GroupReduction([[1, 2, 3, 4]], 16*1024)
        self.assertEqual([[1, 2, 3, 4]], group_reduction.solve_sssg())

    def test_single_switch_single_group_2(self):
        group_reduction = GroupReduction([[20, 40, 60, 80]], 16*1024)
        self.assertEqual([[1, 2, 3, 4]], group_reduction.solve_sssg())

    def test_single_switch_single_group_3(self):
        group_reduction = GroupReduction([[10.5, 20.1, 31.0, 39.7]], 10)
        self.assertEqual([[1, 2, 3, 4]], group_reduction.solve_sssg())

    def test_single_switch_single_group_4(self):
        group_reduction = GroupReduction([[i + 0.1 for i in range(1, 17)]],
                                         16*1024)
        self.assertEqual([[(i + 0.1) * 10 for i in range(1, 17)]],
                         group_reduction.solve_sssg())

    def test_single_switch_multi_group_1(self):
        group_reduction = GroupReduction([[1, 2], [3, 4]], 16*1024)
        self.assertEqual([[1, 2], [3, 4]], group_reduction.solve_ssmg())

    def test_single_switch_multi_group_2(self):
        group_reduction = GroupReduction([[1.1, 2.1], [3.1, 4.1]], 16*1024)
        self.assertEqual([[11, 21], [31, 41]], group_reduction.solve_ssmg())

    def test_single_switch_multi_group_3(self):
        group_reduction = GroupReduction([[7.784, 67.785], [10.753, 14.765]],
                                         16*1024)
        self.assertEqual([[137, 1193], [729, 1001]],
                         group_reduction.solve_ssmg())


if __name__ == "__main__":
    unittest.main()
