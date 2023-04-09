# <img src="./FabricEval-logo.svg" width="30"> FabricEval
FabricEval is a modular evaluation framework built for studying data center
traffic engineering (TE) schemes. FabricEval's end-to-end pipeline includes
three main parts: (1) topology/traffic demand generators, (2) TE solution + implementation,
(3) performance statistics collection.
More specifically, it constructs a data center network topology, overlaid with
a traffic demand matrix for the topology. Then a snapshot is taken on the network
and fed to a TE solver. The output of the TE solver is distributed to each switch
and translated to switch rules in the form of Longest Prefix Match (LPM) flows
and Weighted-Cost Multi-Path (WCMP) groups. Finally, the data plane implementation
is compared to the desired state derived from the original TE solution. A set of
log files will generated to record the precision loss.

FabricEval by default generates production-like data center network topologies,
production-like traffic demand matrices, and uses the same TE algorithm from
Google's Jupiter fabrics. It also models switches with specs, e.g., port speed,
table space, from commodity switch vendors like Broadcom. All of these are
flexible modules, users can easily replace with their favorite configuration and
TE algorithm.

A main component of TE implementation is the group reduction algorithm that
reduces original groups with large sizes to smaller ones to fit into the switch
table limits. FabricEval includes a selection of such group reduction algorithms,
including a re-implementation of WCMP TableFitting \[EuroSys'14\], our own DMIR
and IGR, as well as a few other variants. The group reduction algorithm is also
modular, and can easily be replaced with the user's own choice.

## Structure
* **common/**: common helper functions and flags.
* **e2e/**: a simple pipeline implementation of the entire framework.
* **globalTE/**: traffic engineering solver component.
* **localTE/**: WCMP + group reduction that produces switch rules.
* **proto/**: protobuf definitions for topology, traffic demands, TE solutions and other inputs.
* **scripts/**: log parsing scripts.
* **tests/**: unit tests.
  * **tests/data/**: protobuf format production-like topologies and traffic demand matrices.
* **topology/**: the topology class that represents a network in memory.
* **traffic/**: the traffic class that represents traffic demand matrices.

## Prerequisites
* [Bazel](https://docs.bazel.build/install.html).
* [Python 3](https://www.python.org/downloads/).
* [Gurobi](https://www.gurobi.com/). Everything is only tested on Gurobi 9.5.0+.
* [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/).
* [Protobuf](https://developers.google.com/protocol-buffers).

Make sure the above dependencies are all installed.

## Usage
Run the following command to run all unit tests.
```bash
bazel test //tests:all
```
Run the following command to invoke the e2e pipeline, and dump generated traffic matrix,
TE solution, and link utilization stats to the current path:
```bash
bazel run //e2e:run -- $(pwd)
```
