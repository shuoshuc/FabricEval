---
title: Steps in FabricEval
permalink: /docs/steps/
---

`e2e/run.py` is the starting point of an evaluation run.

#### Step 1
When invoked, step 1 is to generate a network topology. A network topology is a
[Protobuf](https://github.com/protocolbuffers/protobuf)-based representation of
all elements in the network and their relationship. A topology can either be loaded
from a pre-built proto file or generated using FabricEval's builtin topology generator.
More details can be found in the "Topology" section of the documentation.

#### Step 2
Step 2 is to generate traffic demands for the network. With the number of end hosts
known from step 1, users can call FabricEval's builtin traffic matrix generator
to generate demands, or again, load from a proto file. More details can be found
in the "Traffic" section.

#### Step 3
Step 3 is to run the TE solver and obtain a TE solution by calling the GlobalTE
module. The TE solver by default formulates the optimization problem as
minimizing maximum link utilization, i.e., to achieve load balancing.

#### Step 4
In step 4, the TE solution is passed in from the previous stage and prepared for
group reduction. The LocalTE component installs the reduced groups into switches,
where each switch is modeled by a `Node` class (see Topology section). It also
updates the traffic load on each link according to the weight distributions in
each reduced group.

#### Final step(s)
The final step(s) collect various metric from the network, including link utilization,
switch table utilization, admitted demands etc. This can be found in the "Metrics"
section.
