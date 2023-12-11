---
title: Prerequisites
permalink: /docs/home/
redirect_from: /docs/index.html
---

FabricEval has a few dependencies:
- [Python 3](https://www.python.org/downloads/)
- [Bazel](https://bazel.build/install)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Protobuf](https://developers.google.com/protocol-buffers)
- [Gurobi 9.5+](https://www.gurobi.com/)
- [Git](https://git-scm.com/)
- [Git LFS](https://git-lfs.com/)

We need to install the above packages in order to run FabricEval successfully.
Please follow the installation instructions of each package and make sure they
are included into the system path.

A script is provided to test if all the dependencies have been met.
It can be found at `scripts/test_dep.py` in the [FabricEval github repo](https://github.com/shuoshuc/FabricEval).

To use the script, run
```bash
$ python3 ./FabricEval/scripts/test_dep.py
```

If all dependencies are met, the script would output:
> All dependencies are met.

Now it's time to start using FabricEval!
