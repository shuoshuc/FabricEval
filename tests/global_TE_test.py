import unittest

import numpy as np
import proto.te_solution_pb2 as TESolution
from google.protobuf import text_format

from globalTE.global_te import GlobalTE
from topology.topogen import generateToy3, generateToy4
from topology.topology import Topology, filterPathSetWithSeg
from traffic.tmgen import tmgen
from traffic.traffic import Traffic

TOY2_PATH = 'tests/data/toy2.textproto'
TOY2_TRAFFIC_PATH = 'tests/data/toy2_traffic.textproto'

C1AB1 = 'toy2-c1-ab1'
C2AB1 = 'toy2-c2-ab1'
C3AB1 = 'toy2-c3-ab1'
# toy4 entities.
TOY4_C1 = 'toy4-c1-ab1'
TOY4_C2 = 'toy4-c2-ab1'
TOY4_C3 = 'toy4-c3-ab1'
TOY4_C4 = 'toy4-c4-ab1'
TOY4_C5 = 'toy4-c5-ab1'

class TestGlobalTESolution(unittest.TestCase):
    def test_te_sol_toy2(self):
        toy2 = Topology(TOY2_PATH)
        toy2_traffic = Traffic(toy2, TOY2_TRAFFIC_PATH)
        global_te = GlobalTE(toy2, toy2_traffic)
        sol = global_te.solve()
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
        link_util = list(toy2.dumpIdealLinkUtil().items())
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

    def test_te_sol_toy4(self):
        toy4 = Topology('', input_proto=generateToy4())
        traffic_proto = tmgen(tor_level=False,
                              cluster_vector=np.array([1]*5),
                              num_nodes=4,
                              model='single',
                              dist='',
                              netname='toy4')
        toy4_traffic = Traffic(toy4, '', traffic_proto)
        global_te = GlobalTE(toy4, toy4_traffic)
        sol = global_te.solve()
        self.assertEqual(TESolution.TESolution.SolutionType.LEVEL_AGGR_BLOCK,
                         sol.type)
        self.assertEqual(4, len(sol.te_intents))
        for te_intent in sol.te_intents:
            # There should not exist an intent for c5.
            self.assertNotEqual(TOY4_C5, te_intent.target_block)
            if te_intent.target_block == TOY4_C1:
                self.assertEqual(1, len(te_intent.prefix_intents))
                self.assertEqual(TESolution.PrefixIntent.PrefixType.SRC,
                                 te_intent.prefix_intents[0].type)
                # c1->c5 uses VLB-like routing, all paths are used.
                self.assertEqual(16,
                                 len(te_intent.prefix_intents[0].nexthop_entries))
                for nexthop_entry in te_intent.prefix_intents[0].nexthop_entries:
                    self.assertEqual(5000.0, nexthop_entry.weight)
            if te_intent.target_block in [TOY4_C2, TOY4_C3, TOY4_C4]:
                self.assertEqual(1, len(te_intent.prefix_intents))
                self.assertEqual(TESolution.PrefixIntent.PrefixType.TRANSIT,
                                 te_intent.prefix_intents[0].type)
                # Only 4 links in the direct connection to c5 are used.
                self.assertEqual(4,
                                 len(te_intent.prefix_intents[0].nexthop_entries))
                for nexthop_entry in te_intent.prefix_intents[0].nexthop_entries:
                    self.assertEqual(5000.0, nexthop_entry.weight)
        # Verify intended link utilization.
        link_util = toy4.dumpIdealLinkUtil()
        # All links originating from c1 should see 0.125 link util.
        for path_name in toy4.findOrigPathsOfAggrBlock(TOY4_C1).keys():
            for link in toy4.findLinksOfPath(path_name):
                self.assertEqual(0.125, link_util[link.name])
        # All links terminating at c5 should see 0.125 link util.
        for path_name in toy4.findTermPathsOfAggrBlock(TOY4_C5).keys():
            for link in toy4.findLinksOfPath(path_name):
                self.assertEqual(0.125, link_util[link.name])

if __name__ == "__main__":
    unittest.main()
