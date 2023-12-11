---
title: Prerequisites
permalink: /docs/home/
redirect_from: /docs/index.html
---

FabricEval has a few dependencies:
- [Git](https://git-scm.com/)
- [Git LFS](https://git-lfs.com/)
- [Python 3](https://www.python.org/downloads/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Bazel](https://bazel.build/install)
- [Gurobi 9.5+](https://www.gurobi.com/) (license required)

We need to install the above packages in order to run FabricEval successfully.
Please follow the installation instructions of each package and make sure they
are included into the system path.

Note that the rest of the documentation assumes `Ubuntu 20.04+` platforms with `bash`,
but should generally apply to most of the Linux distributions.

A script is provided to test if all the dependencies have been met.
It can be found at `scripts/test_dep.py` in the [FabricEval github repo](https://github.com/shuoshuc/FabricEval).

To use the script, simply run
```bash
$ python3 test_dep.py
```

If all dependencies are met, the script would output:
> All dependencies are met.

Now it's time to start using FabricEval!
