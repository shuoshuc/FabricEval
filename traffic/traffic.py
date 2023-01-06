from dataclasses import dataclass, field
from typing import Dict, Tuple

import proto.traffic_pb2 as traffic
from google.protobuf import text_format


def loadTraffic(filepath):
    if not filepath:
        return None
    demand = traffic.TrafficDemand()
    with open(filepath, 'r', encoding='utf-8') as f:
        text_format.Parse(f.read(), demand)
    return demand

@dataclass
class BlockDemands:
    '''
    A data structure storing all ToR-level demands related to an AggrBlock.
    '''
    # A map of demands where only src belongs to the block.
    src_only: Dict[Tuple[str, str], int]
    # A map of demands where only dst belongs to the block.
    dst_only: Dict[Tuple[str, str], int]
    # A map of demands where both src and dst belong to the block.
    src_dst: Dict[Tuple[str, str], int]

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
        self.demand = {}
        # A map from AggrBlock name to BlockDemands dataclass. Only populated
        # when demand is ToR-level.
        self.demand_by_block = {}
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
                      f'{vol} on {src} => {dst}.')
                return
            # Sanity check: only expects one entry for each src-dst pair.
            if (src, dst) in self.demand:
                print(f'[ERROR] Traffic parsing: found more than 1 entry for '
                      f'{src} => {dst}: {self.demand[(src, dst)]} and {vol}')
                return
            # If ToR demand matrix, finds the parent AggrBlocks so that we can
            # construct a block-level matrix out of it. If 2 ToRs are in the
            # same AggrBlock, the demand is *not* counted towards inter-block
            # demand.
            if is_tor:
                src_aggr_block = self.topo.findAggrBlockOfToR(src).name
                dst_aggr_block = self.topo.findAggrBlockOfToR(dst).name
                src_demands = self.demand_by_block.setdefault(src_aggr_block,
                                                              BlockDemands({},
                                                                           {},
                                                                           {}))
                dst_demands = self.demand_by_block.setdefault(dst_aggr_block,
                                                              BlockDemands({},
                                                                           {},
                                                                           {}))
                # Sanity check: only expects one entry for each src-dst pair.
                if (src, dst) in src_demands.src_only \
                        or (src, dst) in src_demands.src_dst:
                    print(f'[ERROR] Traffic parsing: found more than 1 entry '
                          f'for pair {src} => {dst}: {vol}.')
                    return

                if src_aggr_block != dst_aggr_block:
                    tot = self.demand.setdefault((src_aggr_block,
                                                  dst_aggr_block), 0)
                    self.demand[(src_aggr_block, dst_aggr_block)] = tot + vol
                    src_demands.src_only[(src, dst)] = vol
                    dst_demands.dst_only[(src, dst)] = vol
                else:
                    # If src and dst are in the same block, src_demands and
                    # dst_demands point to the same map.
                    src_demands.src_dst[(src, dst)] = vol
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

    def getBlockDemands(self, block_name):
        '''
        Returns the BlockDemands dataclass for the given block_name. Returns
        None if a block has no demand, which is valid.
        N.B.: should only be called when ToR-level demands exist. If there is
              only AggrBlock-level demands, the entire self.demand_by_block is
              empty.
        '''
        if block_name not in self.demand_by_block:
            return None
        return self.demand_by_block[block_name]
