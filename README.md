# FabricSim
A data center network simulator built for studying traffic engineering schemes.

## Structure
* **common/**: common functions and flags.
* **e2e/**: a simple pipeline implementation of the entire framework.
* **globalTE/**: traffic engineering solver component.
* **localTE/**: the WCMPAlloc class that processes input TE solutions.
* **proto/**: protobuf definitions for topological entities, traffic demands, TE solutions and other inputs.
* **tests/**: unit tests.
* **topology/**: the topology class that represents a network in memory.
* **traffic/**: the traffic class that represents traffic demand matrices.

## Prerequisites
* FabricSim is a Python project, it uses Bazel as its build system.
To start, make sure you have [Bazel](https://docs.bazel.build/install.html) installed.
* Everything is implemented and tested under [Python 3](https://www.python.org/downloads/),
so make sure this is also installed.
* FabricSim uses [Gurobi](https://www.gurobi.com/) in the WCMPAlloc component
to solve group reduction optimization. Everything is only tested on Gurobi 9.5.0+.
* FabricSim also uses [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/).
* FabricSim's input and output are of [Protobuf](https://developers.google.com/protocol-buffers) format,
make sure it is also installed.

## Usage
Run the following command to run all unit tests.
```bash
bazel test //tests:all
```
Run the following command to invoke the e2e pipeline, and dump generated traffic matrix,
globalTE solution, and link utilization stats to the current path:
```bash
bazel run //e2e:run -- $(pwd)
```
