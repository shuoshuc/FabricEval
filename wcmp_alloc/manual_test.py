#!/usr/bin/env python
# -*- coding: utf-8 -*-

from group_reduction import GroupReduction, frac2int_lossless

if __name__ == "__main__":
    input_groups = [[10.5, 20.1, 31.0, 39.7]]
    group_reduction = GroupReduction(input_groups, 16*1024)
    print('Input %s' % input_groups)
    print(group_reduction.solve_sssg())
