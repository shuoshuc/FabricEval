import proto.te_solution_pb2 as te_sol
from google.protobuf import text_format

def loadTESolution(filepath):
    if not filepath:
        return None
    sol = te_sol.TESolution()
    with open(filepath, 'r', encoding='utf-8') as f:
        text_format.Parse(f.read(), sol)
    return sol


class WCMPWorker:
    '''
    A WCMP worker is responsible for mapping the TE intents targeting an
    abstract node that it manages to programmed flows and groups on switches.
    '''
    def __init__(self, topo_obj, te_intent):
        self._target_block = te_intent.target_block
        self._topo = topo_obj
        self._te_intent = te_sol.TEIntent()
        self._te_intent.CopyFrom(te_intent)

    def run(self):
        '''
        Translates the high level TE intents to programmed flows and groups.
        '''
        pass


class WCMPAllocation:
    '''
    WCMP allocation class that handles the intra-cluster WCMP implementation.
    It translates the TE solution to flows and groups that are programmed on
    each switch.
    '''
    def __init__(self, topo_obj, input_proto):
        # A map from cluster name to the corresponding WCMP worker instance.
        self._worker_map = {}
        # Stores the topology object in case we need to look up an element.
        self._topo = topo_obj
        # Loads the full network TE solution.
        proto_sol = loadTESolution(input_proto)
        for te_intent in proto_sol.te_intents:
            aggr_block = te_intent.target_block
            if not topo_obj.hasAggrBlock(aggr_block):
                print('[ERROR] {}: Target block {} does not exist in this '
                      'topology!'.format('Find aggr block', aggr_block))
                continue
            # Here we assume each cluster contains exactly one aggregation
            # block. Since a worker is supposed to align with an SDN control
            # domain, it manages all the aggregation blocks in a cluster. In
            # this case, it only manages one aggregation block.
            self._worker_map[aggr_block] = WCMPWorker(topo_obj, te_intent)

    def run(self):
        for worker in self._worker_map.values():
            worker.run()
