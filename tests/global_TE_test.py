import unittest

import proto.te_solution_pb2 as TESolution
from google.protobuf import text_format

from globalTE.global_te import GlobalTE
from topology.topology import Topology, filterPathSetWithSeg
from traffic.traffic import Traffic

TOY2_PATH = 'tests/data/toy2.textproto'
TOY2_TRAFFIC_PATH = 'tests/data/toy2_traffic.textproto'

C1AB1 = 'toy2-c1-ab1'
C2AB1 = 'toy2-c2-ab1'
C3AB1 = 'toy2-c3-ab1'

class TestGlobalTESolution(unittest.TestCase):
    def test_te_sol_toy2(self):
        toy2 = Topology(TOY2_PATH)
        toy2_traffic = Traffic(toy2, TOY2_TRAFFIC_PATH)
        global_te = GlobalTE(toy2, toy2_traffic)
        sol = global_te.solve()
        print(text_format.MessageToString(sol))
        self.assertEqual(TESolution.TESolution.SolutionType.LEVEL_AGGR_BLOCK,
                         sol.type)
        self.assertEqual(3, len(sol.te_intents))
        for te_intent in sol.te_intents:
            if te_intent.target_block == C1AB1:
                self.assertEqual(1, len(te_intent.prefix_intents))
                self.assertEqual(TESolution.PrefixIntent.PrefixType.SRC,
                                 te_intent.prefix_intents[0].type)
                self.assertEqual(4,
                                 len(te_intent.prefix_intents[0].nexthop_entries))
                for nexthop_entry in te_intent.prefix_intents[0].nexthop_entries:
                    self.assertEqual(75000.0, nexthop_entry.weight)
            if te_intent.target_block == C2AB1:
                self.assertEqual(1, len(te_intent.prefix_intents))
                self.assertEqual(TESolution.PrefixIntent.PrefixType.TRANSIT,
                                 te_intent.prefix_intents[0].type)
                self.assertEqual(2,
                                 len(te_intent.prefix_intents[0].nexthop_entries))
                for nexthop_entry in te_intent.prefix_intents[0].nexthop_entries:
                    self.assertEqual(75000.0, nexthop_entry.weight)
            if te_intent.target_block == C3AB1:
                self.assertEqual(1, len(te_intent.prefix_intents))
                self.assertEqual(TESolution.PrefixIntent.PrefixType.SRC,
                                 te_intent.prefix_intents[0].type)
                self.assertEqual(2,
                                 len(te_intent.prefix_intents[0].nexthop_entries))
                for nexthop_entry in te_intent.prefix_intents[0].nexthop_entries:
                    self.assertEqual(50000.0, nexthop_entry.weight)
        # Verify intended link utilization.
        link_util = list(toy2.dumpLinkUtil().items())
        # 4 src links from C1AB1 have 0.75 link util.
        self.assertEqual(0.75, link_util[0][1])
        self.assertEqual(0.75, link_util[1][1])
        self.assertEqual(0.75, link_util[2][1])
        self.assertEqual(0.75, link_util[3][1])
        # 2 transit links from C2AB1 have 0.75 link util.
        self.assertEqual(0.75, link_util[4][1])
        self.assertEqual(0.75, link_util[5][1])
        # 2 src links from C3AB1 have 0.5 link util.
        self.assertEqual(0.5, link_util[6][1])
        self.assertEqual(0.5, link_util[7][1])

if __name__ == "__main__":
    unittest.main()
