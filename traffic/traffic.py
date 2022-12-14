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
    def __init__(self, topo_obj, input_path, input_proto=None):
        '''
        topo_obj: a topology object matching the input traffic demand.
        input_path (required): path to the textproto of the traffic demand.
        input_proto (optional): raw proto of the traffic demand. 
        '''
        self.topo = topo_obj
        # A map from (s, t) to demand.
        self.demand, self.tor_demand = {}, {}
        # parse input traffic and construct in-mem representation (this class).
        # If a raw proto is given, ignore `input_path`.
        proto_traffic = input_proto if input_proto else loadTraffic(input_path)
        self.demand_type = proto_traffic.type
        is_tor = self.demand_type == traffic.TrafficDemand.DemandType.LEVEL_TOR
        for demand_entry in proto_traffic.demands:
            src, dst = demand_entry.src, demand_entry.dst
            # Sanity check: src and dst cannot be the same.
            if src == dst:
                print(f'[ERROR] Traffic parsing: src {src} and dst {dst} cannot'
                      f' be the same!')
                return
            # Sanity check: only positive demand allowed.
            vol = demand_entry.volume_mbps
            if vol <= 0:
                print(f'[ERROR] Traffic parsing: encountered negative demand: '
                      f'{vol} on s-t {src}-{dst}.')
                return
            # Sanity check: only expects one entry for each src-dst pair.
            if (src, dst) in self.demand:
                print(f'[ERROR] Traffic parsing: found more than 1 entry for '
                      f'pair {src}-{dst}: {self.demand[(src, dst)]} and {vol}')
                return
            if is_tor and (src, dst) in self.tor_demand:
                print(f'[ERROR] Traffic parsing: found more than 1 entry for '
                      f'ToR pair {src}-{dst}: {self.tor_demand[(src, dst)]} and'
                      f'{vol}')
                return
            # If ToR demand matrix, finds the parent AggrBlocks so that we can
            # construct a block-level matrix out of it. If 2 ToRs are in the
            # same AggrBlock, the demand is *not* counted towards inter-block
            # demand.
            if is_tor:
                src_aggr_block = self.topo.findAggrBlockOfToR(src).name
                dst_aggr_block = self.topo.findAggrBlockOfToR(dst).name
                if src_aggr_block != dst_aggr_block:
                    tot = self.demand.setdefault((src_aggr_block,
                                                  dst_aggr_block), 0)
                    self.demand[(src_aggr_block, dst_aggr_block)] = tot + vol
                self.tor_demand[(src, dst)] = vol
            else:
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
    
    def getDemandType(self):
        '''
        Returns the demand type.
        '''
        return self.demand_type
