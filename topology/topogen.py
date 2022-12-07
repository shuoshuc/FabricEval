import proto.topology_pb2 as topo
from google.protobuf import text_format

# Network name
NETNAME = 'toy3'
# Number of gen1/gen2/gen3 clusters.
NGEN1, NGEN2, NGEN3 = 22, 22, 21
# Number of S1/S2/S3 nodes in each cluster.
NS1, NS2, NS3 = 32, 4, 4
# ECMP table limits for each generation.
ECMP_LIMITS = {
    1: 4096,
    2: 16384,
    3: 32768
}
# Number of ports on S1/S2/S3 nodes.
NPORTS1, NPORTS2, NPORTS3 = 32, 128, 128
# Port speeds (in Mbps) for each generation.
PORT_SPEEDS = {
    1: 40000,
    2: 100000,
    3: 200000
}

def getClusterGenByIndex(idx):
    '''
    Returns the generation of a cluster based on the cluster index.
    '''
    if idx < 1 or idx > 65:
        print(f'[ERROR] illegal cluster index {idx}')
        return -1
    if idx <= NGEN1:
        return 1
    if idx <= (NGEN1 + NGEN2):
        return 2
    else:
        return 3

def generateToy3():
    '''
    Generates fabric toy3. Toy3 is a 65-cluster spine-free fabric. It has the
    following spec:
        # 40G (Gen 1) clusters: 22
        # 100G (Gen 2) clusters: 22
        # 200G (Gen 3) clusters: 21
        cluster radix: 256 links
        # AggrBlock per cluster: 1
        # S3 nodes per AggrBlock: 4
        # S2 nodes per AggrBlock: 4
        # S1 nodes (ToR) per AggrBlock: 32
        # ports on S2/S3 nodes: 128
        # ports on S1 nodes: 32
        S1 over-subscription: 1:3
        ECMP table size (40G): 4K
        ECMP table size (100G): 16K
        ECMP table size (200G): 32K

    Returns a populated protobuf-format topology.
    '''
    net = topo.Network()
    net.name = NETNAME
    # Add cluster under network.
    for c_idx in range(1, 66):
        cluster_gen = getClusterGenByIndex(c_idx)
        cluster = net.clusters.add()
        cluster.name = f'{NETNAME}-c{c_idx}'
        # Add AggrBlock under cluster. Each cluster only has 1 AggrBlock.
        aggr_block = cluster.aggr_blocks.add()
        aggr_block.name = f'{cluster.name}-ab1'
        # Add S3 nodes under AggrBlock.
        for s3_idx in range(1, NS3 + 1):
            s3 = aggr_block.nodes.add()
            s3.name = f'{aggr_block.name}-s3i{s3_idx}'
            s3.stage = 3
            s3.index = s3_idx
            s3.ecmp_limit = ECMP_LIMITS[cluster_gen]
            s3.group_limit = 256
            # Add ports under each S3 node.
            for s3p_idx in range(1, NPORTS3 + 1):
                s3_port = s3.ports.add()
                s3_port.name = f'{s3.name}-p{s3p_idx}'
                s3_port.port_speed_mbps = PORT_SPEEDS[cluster_gen]
                # Odd indices indicate DCN facing ports.
                s3_port.dcn_facing = True if s3p_idx % 2 == 1 else False
            # TODO: assign mgmt prefixes.
        # Add S2 nodes under AggrBlock.
        for s2_idx in range(1, NS2 + 1):
            s2 = aggr_block.nodes.add()
            s2.name = f'{aggr_block.name}-s2i{s2_idx}'
            s2.stage = 2
            s2.index = s2_idx
            s2.ecmp_limit = ECMP_LIMITS[cluster_gen]
            s2.group_limit = 256
            # Add ports under each S2 node.
            for s2p_idx in range(1, NPORTS2 + 1):
                s2_port = s2.ports.add()
                s2_port.name = f'{s2.name}-p{s2p_idx}'
                s2_port.port_speed_mbps = PORT_SPEEDS[cluster_gen]
                # S2 nodes have no DCN facing port.
                s2_port.dcn_facing = False
            # TODO: assign mgmt prefixes.

        # Add S1 nodes (ToR) under cluster.
        for s1_idx in range(1, NS1 + 1):
            s1 = cluster.nodes.add()
            s1.name = f'{cluster.name}-ab1-s1i{s1_idx}'
            s1.stage = 1
            s1.index = s1_idx
            s1.ecmp_limit = ECMP_LIMITS[cluster_gen]
            s1.group_limit = 256
            # Add ports under each S1 node.
            for s1p_idx in range(1, NPORTS1 + 1):
                s1_port = s1.ports.add()
                s1_port.name = f'{s1.name}-p{s1p_idx}'
                s1_port.port_speed_mbps = PORT_SPEEDS[cluster_gen]
                # S1 nodes have no DCN facing port.
                s1_port.dcn_facing = False
            # TODO: assign host and mgmt prefixes.

    # Add paths under network.
    for i in range(1, 66):
        for j in range(1, 66):
            # A cluster cannot have a path to itself.
            if i == j:
                continue
            path = net.paths.add()
            path.src_aggr_block = f'{NETNAME}-c{i}-ab1'
            path.dst_aggr_block = f'{NETNAME}-c{j}-ab1'
            path.name = f'{path.src_aggr_block}:{path.dst_aggr_block}'
            i_gen = getClusterGenByIndex(i)
            j_gen = getClusterGenByIndex(j)
            # Path capacity = 4 links * link speed. Link speed is
            # auto-negotiated to be the lower speed between src and dst.
            path.capacity_mbps = 4 * min(PORT_SPEEDS[i_gen], PORT_SPEEDS[j_gen])

    print(text_format.MessageToString(net))
    return net

if __name__ == "__main__":
    generateToy3()
