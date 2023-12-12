---
title: Flags
permalink: /docs/flags/
---

There are many configurable parameters in FabricEval. Most are all defined in
`common/flags.py`, some are defined locally to a file. We go through each of them.

#### Local flags in `e2e/run.py`
`NETWORK`: This local flag specifies the network name to be used
in evaluation. Available values are (detailed spec in `topology/topogen.py`):
| name | size   | topology   |
|------|--------|------------|
| toy3 | large  | spine-free |
| toy4 | small  | spine-free |
| toy5 | medium | spine-free |
| toy6 | large  | Clos       |
| f1   | medium | spine-free |
| f2   | small  | spine-free |

`LOAD_TOPO`: A local flag controlling whether to load a pre-generated
topology from `tests/data/`. The textproto file must exist.

`LOAD_TM`: A local flag controlling whether to load pre-generated
traffic demands from `tests/data/`. The textproto file must exist.

`LOAD_SOL`: A local flag controlling whether to load a pre-generated
TE solution from `tests/data/`. The textproto file must exist.

#### Global flags in `common/flags.py`
`VERBOSE`: The verbosity level of logs. 0 means no informational printing, no Gurobi log.
1 means informational log + Gurobi summary log. 2 means full Gurobi log.

`P_LINK_FAILURE`: Probability of a link failure in the topology. N.B., setting it too high might
cause a network partition.

`EQUAL_INGRESS_EGRESS`: True means the block total ingress should equal its total
egress when generating demands with the traffic demand generator.

`P_SPARSE`: Fraction of blocks with 0 demand in a demand matrix.

`ENABLE_HEDGING`: True to enforce path diversity constraint.

`S`: Spread of path diversity in (0, 1].

`USE_INT_INPUT_GROUPS`: If True, inputs to group reduction algorithms are scaled
up to integer groups (lossless proportional scale up).

`INFINITE_ECMP_TABLE`: True to enable infinite ECMP table size, overrides
`TABLE_LIMIT` and `MAX_GROUP_SIZE` flag.

`TABLE_LIMIT`: Switch table limit.

`MAX_GROUP_SIZE`: Max ECMP entries a group is allowed to use. Some group reduction
algorithms might enforce a per-group limit.

`IMPROVED_HEURISTIC`: True to enable a set of improved heuristics in group reduction.
(1) pruning policy. (2) max group size. (3) table limit used. (4) group admission policy.

`EUROSYS_MOD`: True to enable modified EuroSys algorithm, i.e., performs pruning.

`PARALLELISM`: Number of parallel group reductions allowed to run.

`GUROBI_TIMEOUT`: Timeout in seconds for a single Gurobi invocation.

`GR_ALGO`: The algorithm to use for group reduction. Must be one of
\{eurosys, eurosys_mod, google, igr, dmir, gurobi\}.

`DUMP_GROUPS`: True to dump groups from switches to a csv file.
