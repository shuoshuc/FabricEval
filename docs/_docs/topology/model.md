---
title: Model
permalink: /docs/model/
---

The network is modeled as a [multi-aggregation-layer representation](https://research.google/pubs/pub48991/)
using [Protobuf](https://github.com/protocolbuffers/protobuf).

Here is a block diagram of the elements in a network topology:

                       +----------------------------------+
                       |               spine              |
                       |                                  |
                       |      +----+          +----+      |
                       |      | S5 |          | S5 |      |
                       |      +----+          +----+      |
                       |  +----+  +----+  +----+  +----+  |
                       |  | S4 |  | S4 |  | S4 |  | S4 |  |
                       |  +----+  +----+  +----+  +----+  |
                       +----------------------------------+

                      +------------------------------------+
                      |               Cluster              |
                      |                                    |
                      |   +-----------------------------+  |
                      |   |      Aggregation block      |  |
                      |   |                             |  |
                      |   | +----+ +----+ +----+ +----+ |  |
                      |   | | S3 | | S3 | | S3 | | S3 | |  |
                      |   | +----+ +----+ +----+ +----+ |  |
                      |   | +----+ +----+ +----+ +----+ |  |
                      |   | | S2 | | S2 | | S2 | | S2 | |  |
                      |   | +----+ +----+ +----+ +----+ |  |
                      |   +-----------------------------+  |
                      |                                    |
                      | +----+ +----+ +----+ +----+ +----+ |
                      | | S1 | | S1 | | S1 | | S1 | | S1 | |
                      | +----+ +----+ +----+ +----+ +----+ |
                      +------------------------------------+

There are clusters, which consist of an aggregation block switch and a set of stage
1/ToR (*S1*) switches. Inside each aggregation block, there are 4 stage 2 (*S2*) switches
and 4 stage 3 (*S3*) switches.

Each S1 has a total 16 or 32 links. With a typical 1:3 oversubscription ratio,
8 links are facing the S2, each S2 gets 2 incoming links. Each S2 and S3 have 32
or 64 total links, half facing south and half facing north. The links on S2 are
even distributed to all S3 switches.

Depending on whether the topology is Clos-based or spine-free, there may or may
not be a spine block of *S4* and *S5* switches. If spine-free, the clusters are
directly connected to all others as a full mesh.

The complete definition of network entities can be found in `proto/topology.proto`.
We paste it here:

```protobuf
message Port {
    // Unique name identifier.
    string name = 1;
    // Port speed in Mbits/sec.
    int64 port_speed_mbps = 2;
    // True if this port is facing the data-center network (DCN).
    bool dcn_facing = 3;
    // True if this port is facing the hosts.
    bool host_facing = 4;
    // Unique index among the ports of the same nodes.
    int32 index = 5;
}

message Link {
    // Unique name identifier.
    string name = 1;
    // Unique name identifier of the source port.
    string src_port_id = 2;
    // Unique name identifier of the destination port.
    string dst_port_id = 3;
    // Link speed in Mbits/sec.
    int64 link_speed_mbps = 4;
}

message Node {
    // Unique name identifier.
    string name = 1;
    // Stage of the node, 1 for ToR, 2/3 for AggregationBlock.
    int32 stage = 2;
    // Unique index among the nodes of the same stage.
    int32 index = 3;
    // Number of LPM entries the flow table can hold.
    int64 flow_limit = 4;
    // Number of ECMP entries the ECMP table can hold.
    int64 ecmp_limit = 5;
    // Max number of ECMP entries each group can use.
    int64 group_limit = 6;
    // Member ports on the node.
    repeated Port ports = 7;
    // Aggregated IPv4 prefix of all hosts in the rack (only a ToR can have the
    // prefix fields set).
    string host_prefix = 8;
    // Netmask of the host_prefix in slash notation.
    uint32 host_mask = 9;
    // Assigned IPv4 management prefix for a ToR (used for out-of-band
    // management connection in an SDN network).
    string mgmt_prefix = 10;
    // Netmask of the mgmt_prefix in slash notation.
    uint32 mgmt_mask = 11;
}

message AggregationBlock {
    // Unique name identifier.
    string name = 1;
    // Member nodes in the aggregation block.
    repeated Node nodes = 2;
}

message Path {
    // Unique name identifier.
    string name = 1;
    // Unique name identifier of the source aggregation block.
    string src_aggr_block = 2;
    // Unique name identifier of the destination aggregation block.
    string dst_aggr_block = 3;
    // Path capacity in Mbits/sec, should match the sum of member link capacity.
    int64 capacity_mbps = 4;
}

message Cluster {
    // Unique name identifier.
    string name = 1;
    // Member aggregation blocks in the cluster.
    repeated AggregationBlock aggr_blocks = 2;
    // Member nodes directly belonging to the cluster (e.g., ToRs).
    repeated Node nodes = 3;
}

message Network {
    // Unique name identifier.
    string name = 1;
    // Member clusters in the network.
    repeated Cluster clusters = 2;
    // Member paths in the network.
    repeated Path paths = 3;
    // Member links in the network.
    repeated Link links = 4;
}
```
