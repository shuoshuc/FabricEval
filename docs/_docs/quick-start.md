---
title: Quick start
permalink: /docs/quick-start/
---

Now let's get the source code of FabricEval from GitHub:
```bash
$ git clone https://github.com/shuoshuc/FabricEval.git
```

We can run the end-to-end FabricEval pipeline:
```bash
$ bazel run //e2e:run -- $(pwd)
```

The command takes less than a minute to complete on a 16-core machine.
