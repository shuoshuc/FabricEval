# FabricSim
A data center network simulator designed for studying traffic engineering schemes.

## Structure
* **proto/**: protobuf definitions for topological entities, traffic demands and other inputs.
* **topology/**: the topology class that represents a network in memory.
* **tests/**: unit tests.

## Prerequisites
FabricSim is a Python project, it uses Bazel as its build system.
To start, make sure you have [Bazel](https://docs.bazel.build/install.html) installed.
Everything is implemented and tested under [Python 3](https://www.python.org/downloads/), so make sure this is also installed.

## Usage
Run the following command to unit test loading a test network topology and verify its properties:
```bash
bazel test //tests:load_toy
```
