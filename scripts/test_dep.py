#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests whether all dependencies are met.
# $ python3 test_dep.py

import subprocess
import sys

import gurobipy as gp
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
    if gp.gurobi.version() < (9, 5, 0):
        print("ERROR: Gurobi version below 9.5.0")
        return
    gurobi_license = str(subprocess.run(['gurobi_cl', '--license'], capture_output=True).stdout)
    if 'expired' in gurobi_license:
        print("ERROR: Gurobi license expired")
        return
    git_ver = str(subprocess.run(['git', '--version'], capture_output=True).stdout)
    if 'git version' not in git_ver:
        print("ERROR: could not find git")
        return
    gitlfs_ver = str(subprocess.run(['git', 'lfs', '--version'], capture_output=True).stdout)
    if 'git-lfs' not in gitlfs_ver:
        print("ERROR: could not find git lfs")
        return
    bazel_ver = str(subprocess.run(['bazel', '--version'], capture_output=True).stdout)
    if 'bazel' not in bazel_ver:
        print("ERROR: could not find bazel")
        return
    print("All dependencies are met.")

if __name__ == "__main__":
    test_dep()
