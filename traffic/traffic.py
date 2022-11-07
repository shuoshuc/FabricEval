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
    Traffic class that represents the demand matric of a network. It contains
    ToR-level demand and/or aggregation-block-level demand.
    '''
    def __init__(self, input_proto):
        pass
