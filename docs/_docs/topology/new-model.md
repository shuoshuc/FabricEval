---
title: Create a new model
permalink: /docs/new-model/
---

FabricEval has builtin topology generator that takes in a high-level spec of the
desired network and outputs a complete protobuf. It can be found at `topology/topogen.py`.

For example, a desired topology might have the following high-level spec:

```
Number of 40G (Gen 1) clusters: 22
Number of 100G (Gen 2) clusters: 22
Number of 200G (Gen 3) clusters: 21
cluster radix: 256 links
Number of AggrBlock per cluster: 1
Number of S3 nodes per AggrBlock: 4
Number of S2 nodes per AggrBlock: 4
Number of S1 nodes (ToR) per AggrBlock: 32
Number of ports on S2/S3 nodes: 128
Number of ports on S1 nodes: 32
S1 over-subscription: 1:3
ECMP table size (40G): 4K
ECMP table size (100G): 16K
ECMP table size (200G): 32K
```

The corresponding parameters are set in `generateToy3()`. Briefly speaking, the
topology generator constructs clusters one-by-one, then it calls a `StripingPlan`
class, which connects the clusters to each other in a balanced way (minimal difference
in the number of links).

Users can build a new topology starting from copying the existing topology and
then make modifications on top of it. Another low level approach to construct/modify
a topology is to directly edit the protobuf file.
