# <img src="./FabricEval-logo.svg" width="30"> FabricEval
FabricEval is a modular evaluation framework built for studying traffic
engineering (TE) schemes. FabricEval works as follows:
First, it constructs a data center network topology, overlaid with
a traffic demand matrix for the topology. Then a snapshot is taken on the network
and fed to a TE solver. The output of the TE solver is distributed to each switch
and translated to switch rules in the form of Weighted-Cost Multi-Path (WCMP) groups.
Finally, the data plane implementation is compared to the desired state derived
from the original TE solution. A set of log files will be generated to record the precision loss.

FabricEval by default generates production-like data center network topologies/traffic
demand matrices from Google's [Jupiter fabrics](https://research.google/pubs/pub51587/),
and uses the same TE algorithm powering the
[Orion SDN controller](https://research.google/pubs/pub50245/). It also models
switches with specs, e.g., port speed, table space, from commodity switch vendors
like Broadcom. All of these are flexible modules, users can easily replace with
their favorite configuration and TE algorithm.

A main component of TE implementation is the group reduction algorithm that
reduces original groups with large sizes to smaller ones to fit into the switch
table limits. FabricEval includes a selection of such group reduction algorithms,
including a re-implementation of WCMP TableFitting \[EuroSys'14\], our own DMIR
and IGR algorithms, as well as a few other variants. The group reduction algorithm
is also modular, and can easily be replaced with the user's own choice.

## Structure
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

## Prerequisites
* [Bazel 7.0.0+](https://docs.bazel.build/install.html).
* [Python 3.8+](https://www.python.org/downloads/).
* [Gurobi](https://www.gurobi.com/). Everything is only tested on Gurobi 9.5.0+.
* [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/).

Make sure the above dependencies are all installed.

## Usage
Run the following command to run all unit tests.
```bash
bazel test //tests:all
```
Run the following command to invoke the e2e pipeline, and dump generated traffic matrix,
TE solution, and link utilization stats to `igr/` in the current path:
```bash
bazel run //e2e:run -- $(pwd)
```

More details can be found [here](https://shuoshuc.github.io/FabricEval/).
