---
title: Traffic matrix
permalink: /docs/tm/
---

Each network topology is associated with a traffic demand matrix, represented as
a protobuf:

```protobuf
message DemandEntry {
    // Unique name identifier of source (ToR or aggregation block).
    string src = 1;
    // Unique name identifier of destination (ToR or aggregation block).
    string dst = 2;
    // Traffic volume in Mbits/sec.
    uint64 volume_mbps = 3;
}

message TrafficDemand {
    enum DemandType {
      LEVEL_UNKNOWN = 0; // Unknown as the default forces explicitly set type.
      LEVEL_TOR = 1; // ToR-level demand.
      LEVEL_AGGR_BLOCK = 2; // Aggregation-block-level demand.
    }
    // Specifies demand as one of the enums.
    DemandType type = 1;
    // A traffic demand can either be a ToR-to-ToR demand or an aggregated one.
    // But all repeated entries must be consistent.
    repeated DemandEntry demands = 2;
}
```

Traffic demand matrix comes in two flavors, either at the ToR-level granularity
or at the aggregation-block-level granularity. For ToR-level demands, FabricEval
aggregates them into aggregation-block-level demands for GlobalTE to consume.
LocalTE on the other hand, prefers ToR-level demands to achieve finer-grained
evaluation results.

Based on measurements from Google's production fabrics, the traffic demands
follow a [gravity model](https://research.google/pubs/pub51587/). FabricEval
implements a `tmgen()` in `traffic/tmgen.py` to generate demands following the
gravity model. It can also generate demands following Pareto, exponential and
uniform distributions.

In addition, this traffic demand generator follows practical production fabrics
by introducing flag: `EQUAL_INGRESS_EGRESS` to allow asymmetric ingress/egress,
`P_SPARSE ` to allow a number of cluters to be in expansion mode and not carry
traffic, `P_SPIKE` to approximate instantaneous bursts.

Pre-generated traffic demands for fabrics listed in the "Topology" section
can be found in `tests/data/`. Due to Github's storage size limit, demands for
larger networks are not provided, we recommend generating them in runtime.
