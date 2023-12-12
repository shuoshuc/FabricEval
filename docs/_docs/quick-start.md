---
title: Quick start
permalink: /docs/quick-start/
---

Let's get the source code of FabricEval from GitHub:
```bash
$ git clone https://github.com/shuoshuc/FabricEval.git
```

We can run the end-to-end FabricEval pipeline:
```bash
$ cd FabricEval
$ bazel run //e2e:run -- $(pwd)
```

The command takes less than a minute to complete on a 16-core machine.
We can also see the output of each step in stdout. After the run returns, we
should find a folder named `google_new` in the top level of the repo,
with a set of csv files. This contains all the metrics collected.

For example, `google_new/LU.csv` is a file containing the utilization of all links
in the network. For each link, an ideal utilization given by the TE solution (if no precision loss)
and an actual utilization given by the TE implementation are logged.

In addition, `google_new/node_ecmp.csv` contains the group/ECMP table usage of
every switch, and `google_new/node_demand.csv` contains the admitted traffic demand volume
of each switch.

## Code structure
As mentioned, FabricEval has multiple modules. Below is the structure:
* **common/**: common helper functions and flags.
* **e2e/**: pipeline implementation of the entire framework.
* **globalTE/**: traffic engineering solver component, generates TE solution.
* **localTE/**: handles TE solution to TE implementation mapping + group reduction.
* **proto/**: protobuf definitions for topology, traffic demands, TE solution and so on.
* **scripts/**: log parsing scripts.
* **tests/**: unit tests.
  * **tests/data/**: protobuf format production-like network configs.
* **topology/**: the topology component that represents a network in memory.
* **traffic/**: the traffic component that represents traffic demands.

We have dedicated sections in the rest of the documentation, describing each of
the following components: e2e, topology, traffic, globalTE, localTE.
