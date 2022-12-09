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
    if idx < 1 or idx > (NGEN1 + NGEN2 + NGEN3):
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
    for c_idx in range(1, NGEN1 + NGEN2 + NGEN3 + 1):
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
    for i in range(1, NGEN1 + NGEN2 + NGEN3 + 1):
        for j in range(1, NGEN1 + NGEN2 + NGEN3 + 1):
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

    # Add links under network.
    for c_idx in range(1, NGEN1 + NGEN2 + NGEN3 + 1):
        cluster_gen = getClusterGenByIndex(c_idx)
        # Fetch the only AggrBlock in each cluster.
        aggr_block = net.clusters[c_idx - 1].aggr_blocks[0]
        for node in aggr_block.nodes:
            # ===== Add links between S1 and S2 nodes. =====
            if node.stage == 2:
                # Hard-coded S1-S2 striping: each S2 spreads 64 links to 32 S1
                # nodes, so 2 links per S1. The 2 ports of the links on S2 side
                # use port indices 64 apart. Each S1 uses the first 8 ports to
                # connect to S2.
                for t_idx in range(1, NS1 + 1):
                    # Add first S2->S1 link.
                    link1_s2_s1 = net.links.add()
                    src_port_id = f'{node.name}-p{t_idx * 2}'
                    dst_port_id = f'{aggr_block.name}-s1i{t_idx}-p{node.index}'
                    link1_s2_s1.src_port_id = src_port_id
                    link1_s2_s1.dst_port_id = dst_port_id
                    link1_s2_s1.name = f'{src_port_id}:{dst_port_id}'
                    link1_s2_s1.link_speed_mbps = PORT_SPEEDS[cluster_gen]
                    # Add first S1->S2 link.
                    link2_s1_s2 = net.links.add()
                    src_port_id = f'{aggr_block.name}-s1i{t_idx}-p{node.index}'
                    dst_port_id = f'{node.name}-p{t_idx * 2}'
                    link2_s1_s2.src_port_id = src_port_id
                    link2_s1_s2.dst_port_id = dst_port_id
                    link2_s1_s2.name = f'{src_port_id}:{dst_port_id}'
                    link2_s1_s2.link_speed_mbps = PORT_SPEEDS[cluster_gen]
                    # Add second S2->S1 link.
                    link3_s2_s1 = net.links.add()
                    src_port_id = f'{node.name}-p{t_idx * 2 + int(NPORTS2 / 2)}'
                    dst_port_id = f'{aggr_block.name}-s1i{t_idx}-p{node.index + NS2}'
                    link3_s2_s1.src_port_id = src_port_id
                    link3_s2_s1.dst_port_id = dst_port_id
                    link3_s2_s1.name = f'{src_port_id}:{dst_port_id}'
                    link3_s2_s1.link_speed_mbps = PORT_SPEEDS[cluster_gen]
                    # Add second S1->S2 link.
                    link4_s1_s2 = net.links.add()
                    src_port_id = f'{aggr_block.name}-s1i{t_idx}-p{node.index + NS2}'
                    dst_port_id = f'{node.name}-p{t_idx * 2 + int(NPORTS2 / 2)}'
                    link4_s1_s2.src_port_id = src_port_id
                    link4_s1_s2.dst_port_id = dst_port_id
                    link4_s1_s2.name = f'{src_port_id}:{dst_port_id}'
                    link4_s1_s2.link_speed_mbps = PORT_SPEEDS[cluster_gen]
            if node.stage == 3:
                # ===== Add links between S2 and S3 nodes. =====
                for s2_idx in range(1, NS2 + 1):
                    # Hard-coded S2-S3 striping: the first 16 ports of an S3 are
                    # connected to the first S2 node, the next 16 ports of an S3
                    # are connected to the next S2 node, etc. The 16 ports of
                    # the first S3 node are connected to the first 16 ports of
                    # each S2. The next 16 ports are connected to the next 16
                    # ports of the S2, etc.
                    for i, j in zip(range(32 * (s2_idx - 1) + 2,
                                          32 * s2_idx + 1,
                                          2),
                                    range(32 * node.index - 31,
                                          32 * node.index,
                                          2)):
                        # Add S3->S2 link.
                        link_s3_s2 = net.links.add()
                        src_port_id = f'{node.name}-p{i}'
                        dst_port_id = f'{aggr_block.name}-s2i{s2_idx}-p{j}'
                        link_s3_s2.src_port_id = src_port_id
                        link_s3_s2.dst_port_id = dst_port_id
                        link_s3_s2.name = f'{src_port_id}:{dst_port_id}'
                        link_s3_s2.link_speed_mbps = PORT_SPEEDS[cluster_gen]
                        # Add S2->S3 link.
                        link_s2_s3 = net.links.add()
                        src_port_id = f'{aggr_block.name}-s2i{s2_idx}-p{j}'
                        dst_port_id = f'{node.name}-p{i}'
                        link_s2_s3.src_port_id = src_port_id
                        link_s2_s3.dst_port_id = dst_port_id
                        link_s2_s3.name = f'{src_port_id}:{dst_port_id}'
                        link_s2_s3.link_speed_mbps = PORT_SPEEDS[cluster_gen]
                # ===== Add S3-S3 inter-cluster links. =====
                # Calculate peer port index.
                peer_i = 2 * (c_idx - 1) + 1
                # Hard-coded striping: S3 nodes of the same index connect to
                # each other. S3 of C1 spreads its links to port 1 of all peers.
                # S3 of C2 spreads its N-1 links (first link already connects to
                # C1) to port 3 of rest peers. S3 of C3 spreads N-2 links to
                # port 5 of rest peers, etc.
                for port_idx in range(peer_i, NPORTS3, 2):
                    # Calculate peer cluster index.
                    peer_c_idx = int(port_idx // 2 + 2)
                    peer_aggr_block = f'{NETNAME}-c{peer_c_idx}-ab1'
                    speed = min(PORT_SPEEDS[cluster_gen],
                                PORT_SPEEDS[getClusterGenByIndex(peer_c_idx)])
                    # Add current S3 to peer S3 link.
                    link_away = net.links.add()
                    src_port_id = f'{node.name}-p{port_idx}'
                    dst_port_id = f'{peer_aggr_block}-s3i{node.index}-p{peer_i}'
                    link_away.src_port_id = src_port_id
                    link_away.dst_port_id = dst_port_id
                    link_away.name = f'{src_port_id}:{dst_port_id}'
                    link_away.link_speed_mbps = speed
                    # Add peer S3 to current S3 link.
                    link_back = net.links.add()
                    src_port_id = f'{peer_aggr_block}-s3i{node.index}-p{peer_i}'
                    dst_port_id = f'{node.name}-p{port_idx}'
                    link_back.src_port_id = src_port_id
                    link_back.dst_port_id = dst_port_id
                    link_back.name = f'{src_port_id}:{dst_port_id}'
                    link_back.link_speed_mbps = speed

    return net

if __name__ == "__main__":
    net_proto = generateToy3()
    print(text_format.MessageToString(net_proto))
