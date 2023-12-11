#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests whether all dependencies are met.
# $ python3 test_dep.py

import subprocess
import sys

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
    git_ver = str(subprocess.run(['git', '--version'], capture_output=True).stdout)
    if 'git version' not in git_ver:
        print("ERROR: could not find git")
        return
    bazel_ver = str(subprocess.run(['bazel', '--version'], capture_output=True).stdout)
    if 'bazel' not in bazel_ver:
        print("ERROR: could not find bazel")
        return
    print("All dependencies are met.")

if __name__ == "__main__":
    test_dep()
