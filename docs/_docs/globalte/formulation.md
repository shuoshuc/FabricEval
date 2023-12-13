---
title: Fomulation
permalink: /docs/formulation/
---

The GlobalTE component solves a large-scale TE optimization problem. For better
scalability, GlobalTE performs a two-level hierarchical solving.

As the traffic demands are aggregated to an aggregation-block-level matrix,
GlobalTE solves the TE problem on the aggregation block level, i.e., it finds
the optimal traffic placement between aggregation block pairs. Then the TE solution
for each demand is grouped together by clusters. LocalTE takes the snippet of TE
solution for a cluster and further converts that solution into detailed physical
switch level TE solution. LocalTE will be introduced in the next section.

Assuming a network as a graph $G = (V, E)$, the aggregation block level TE
formulation in GlobalTE is as follows:
```math
minimize u_{max}
s.t. u(x, y) \le u_{max}, \forall (x, y) \in E
```
