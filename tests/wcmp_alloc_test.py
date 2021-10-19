from wcmp_alloc.wcmp_alloc import loadTESolution, WCMPAllocation
import unittest

TOY2_SOL_PATH = 'tests/data/te_sol_toy2.textproto'

class TestWCMPAlloc(unittest.TestCase):
    def test_load_invalid_te_solution(self):
        self.assertEqual(None, loadTESolution(''))

    def test_load_valid_te_solution(self):
        '''
        sol = loadTESolution(TOY2_SOL_PATH)
        self.assertNotEqual(None, sol)
        '''

    def test_toy2_sol_entries(self):
        '''
        sol1 = loadTESolution(SOL1_PATH)
        # check # TEIntents
        self.assertEqual(1, len(sol1.te_intents))
        aggr_block_set = set()
        for te_intent in sol1.te_intents:
            aggr_block_set.add(te_intent.target_block)
        self.assertEqual(1, len(aggr_block_set))
        '''
        pass

if __name__ == "__main__":
    unittest.main()
