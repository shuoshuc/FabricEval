---
title: Naming
permalink: /docs/naming/
---

Entities in the network topology follow this naming scheme:
\{network name\}-\{cluster name\}-\{aggregation block name\}-\{switch stage\}\{switch name\}-\{port name\}

For example, "toy4-c1-ab1-s3i1-p1" is the name of a port. "toy4" is the network
name, "c1" is the cluster name, "ab1" is the aggregation block name, "s3i1" means
a stage 3 switch named i1, "p1" is the port name on the switch.

Similarly, "toy4-c1-ab1-s3i1" is the name of a switch (port name dropped), and
"toy4-c1" is the name of a cluster.

Connections follow this naming scheme:
\{entity name\}:\{entity name\}

For example, "toy4-c3-ab1-s2i2-p5:toy4-c3-ab1-s3i3-p4" is the name of a link.
The originating port of the link is "toy4-c3-ab1-s2i2-p5", the terminating port
is "toy4-c3-ab1-s3i3-p4". We also see that the links are unidirectional in the
topology, which means there is another link "toy4-c3-ab1-s3i3-p4:toy4-c3-ab1-s2i2-p5"
in the reverse direction.

Similarly, "toy4-c3-ab1:toy4-c1-ab1" is a path (a bundle of parallel links) between
"toy4-c3-ab1" and "toy4-c1-ab1".
