# FabricSim
A data center network simulator built for studying traffic engineering schemes.

## Structure
* **proto/**: protobuf definitions for topological entities, traffic demands, TE solutions and other inputs.
* **tests/**: unit tests.
* **topology/**: the topology class that represents a network in memory.
* **wcmp\_alloc/**: the WCMPAlloc class that processes input TE solutions.

## Prerequisites
* FabricSim is a Python project, it uses Bazel as its build system.
To start, make sure you have [Bazel](https://docs.bazel.build/install.html) installed.
* Everything is implemented and tested under [Python 3](https://www.python.org/downloads/),
so make sure this is also installed.
* FabricSim uses [Gurobi](https://www.gurobi.com/) in the WCMPAlloc component
to solve group reduction optimization. Everything is only tested on Gurobi 9.5.0+.

## Usage
Run the following command to unit test loading a test network topology and verify its properties:
```bash
bazel test //tests:load_toy_test
```
Run the following command to unit test calling WCMPAlloc to process a TE solution:
```bash
bazel test //tests:wcmp_alloc_test
```
Run the following command to invoke the e2e pipeline:
```bash
bazel run //e2e:run
```
