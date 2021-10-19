from wcmp_alloc.wcmp_alloc import loadTESolution, WCMPAllocation
import unittest
import ipaddress

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

if __name__ == "__main__":
    unittest.main()
