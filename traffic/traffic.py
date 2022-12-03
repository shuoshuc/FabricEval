import proto.traffic_pb2 as traffic
from google.protobuf import text_format


def loadTraffic(filepath):
    if not filepath:
        return None
    demand = traffic.TrafficDemand()
    with open(filepath, 'r', encoding='utf-8') as f:
        text_format.Parse(f.read(), demand)
    return demand

class Traffic:
    '''
    Traffic class that represents the demand matrix of a network. It contains
    ToR-level demand and/or aggregation-block-level demand.
    '''
    def __init__(self, input_proto):
        '''
        input_proto: textproto format of the traffic demand.
        '''
        # A map from (s, t) to demand.
        self.demand = {}
        # parse input traffic and construct in-mem representation (this class).
        proto_traffic = loadTraffic(input_proto)
        # TODO: flag is_tor_level?
        for demand_entry in proto_traffic.demands:
            src, dst = demand_entry.src, demand_entry.dst
            # Sanity check: src and dst cannot be the same.
            if src == dst:
                print('[ERROR] Traffic parsing: src {} and dst {} cannot be the'
                      ' same!'.format(src, dst))
                return
            # Sanity check: only positive demand allowed, for proto efficiency.
            vol = demand_entry.volume_mbps
            if vol <= 0:
                print('[ERROR] Traffic parsing: encountered negative demand: {}'
                      ' on src-dst {}-{}.'.format(vol, src, dst))
                return
            # Sanity check: only expects one entry for each src-dst pair.
            if (src, dst) in self.demand:
                print('[ERROR] Traffic parsing: found more than 1 entry for '
                      'src-dst pair {}-{}: {} and {}'.format(src, dst,
                                                             self.demand[(src,
                                                                          dst)],
                                                             vol))
                return
            self.demand[(src, dst)] = vol

    def getAllDemands(self):
        '''
        Returns the whole network traffic demand.
        '''
        return self.demand

    def getDemand(self, src, dst):
        '''
        Returns a single demand for (src, dst).
        '''
        return self.demand[(src, dst)]
