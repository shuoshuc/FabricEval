---
title: Group reduction
permalink: /docs/group-reduction/
---

FabricEval implements various group reduction algorithms in `localTE/group_reduction.py`.
This includes prior work *WCMP TableFitting* ([EuroSys 14](https://dl.acm.org/doi/abs/10.1145/2592798.2592803)),
iterative greedy reduction (*IGR*), direct mixed-integer reduction (*DMIR*), Google's
variant of *WCMP TableFitting*, and monolithic Gurobi-based reduction etc.

Users can specify the group reduction algorithm to use by setting flag `GR_ALGO`.
It is also possible to introduce a new algorithm by registering it in `GroupReduction.solve()`.
One thing to keep in mind is to maintain a consistent structure of input and output
groups. Namely, each group in the output is 1:1 mapped to the group at the same
index in the input.

Also note that *DMIR* is implemented using Gurobi-based optimization. `FLAG.GUROBI_TIMEOUT`
sets the timeout duration to be 120 seconds. Sometimes, a potential solution is
not found because the optimization is cut off early. Increasing the timeout might
help in such a case. Besides, users should keep an eye out for memory usage, some
formulation for large scale networks could involve a lot of decision variables
and constraints, thus consuming a lot of memory.
