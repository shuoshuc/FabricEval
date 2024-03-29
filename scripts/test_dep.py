#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests whether all dependencies are met.
# $ python3 test_dep.py

import re
import subprocess
import sys

import gurobipy as gp
import neo4j as n4j
import numpy as np
import scipy as sp


def str2tup(string):
    return tuple(map(int, string.split('.')))


def test_dep():
    if sys.version_info.major < 3 or sys.version_info.minor < 8:
        print("ERROR: python version below 3.8")
        return
    if str2tup(np.__version__) < (1, 18, 0):
        print("ERROR: numpy version below 1.18.0")
        return
    if str2tup(sp.__version__) < (1, 8, 0):
        print("ERROR: scipy version below 1.8.0")
        return
    if str2tup(n4j.__version__) < (5, 0, 0):
        print("ERROR: Neo4j version below 5.0.0")
        return
    if gp.gurobi.version() < (9, 5, 0):
        print("ERROR: GurobiPy version below 9.5.0")
        return
    gurobi_ver = str(
        subprocess.run(['gurobi_cl', '--version'], capture_output=True).stdout)
    m = re.search(r"version (\d+)\.(\d+)\.(\d+) build", gurobi_ver)
    if (int(m[1]), int(m[2]), int(m[3])) < (9, 5, 0):
        print("ERROR: Gurobi version below 9.5.0")
        return
    gurobi_license = str(
        subprocess.run(['gurobi_cl', '--license'], capture_output=True).stdout)
    if 'Error' in gurobi_license:
        print("ERROR: no valid Gurobi license")
        return
    git_ver = str(
        subprocess.run(['git', '--version'], capture_output=True).stdout)
    if 'git version' not in git_ver:
        print("ERROR: could not find git")
        return
    bazel_ver = str(
        subprocess.run(['bazel', '--version'], capture_output=True).stdout)
    g = re.search(r"bazel (\d+)\.(\d+)\.(\d+)", bazel_ver)
    if (int(g[1]), int(g[2]), int(g[3])) < (7, 0, 0):
        print("ERROR: bazel version below 7.0.0")
        return
    print("All dependencies are met.")


if __name__ == "__main__":
    test_dep()
