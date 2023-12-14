---
title: Interact with the topology
permalink: /docs/interaction/
---

FabricEval supports interactive query and visualization of the network topology.
A topology can be loaded into the [Neo4j](https://neo4j.com/) graph database if
`ENABLE_GRAPHDB` is set to True.
Users can visualize the connection and network structure as a graph.

To interact with the network, users can use the [Cypher query language](https://neo4j.com/docs/cypher-manual/current/introduction/).
For instance, users can: retrieve the ports of a switch, find the peer port of a
port, obtain all links between two clusters etc.

#### How to use Neo4j and Cypher
When graph DB is enabled, FabricEval constructs a topology graph in Neo4j while
it builds/parses the generated topology from protobuf. We need to make sure the
backend Neo4j instance is up and running, e.g., at URI "bolt://localhost:7687".

We recommend using a Docker instance, please read more about running Neo4j
[here](https://neo4j.com/docs/operations-manual/current/docker/introduction/).

Next step is to invoke the normal FabricEval run and wait for it to complete.
At this point, a populated topology should exist in Neo4j. We can go to the Neo4j
web server for visual interaction. Just open a brower and visit "http://<neo4j instance ip>:7474".
If the docker instance has a username and password set, it will prompt.

After getting in, we can start writing Cypher queries to retrieve useful information.
For example, let's find the member switches in aggregation block "f2-c3-ab1".
The query returns a graph of 8 switches and 1 aggregation block, with the relationship
plotted as arrows. We can see that this aggregation block contains 8 member switches,
4 of stage 2 and 4 of stage 3.

![screenshot](/assets/img/neo4j.png)
