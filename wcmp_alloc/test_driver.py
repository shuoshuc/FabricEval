#!/usr/bin/env python
# -*- coding: utf-8 -*-

from group_reduction import *
import csv

if __name__ == "__main__":
    table_limit = 16*1024
    # groups, # port per group, lower bound, upper bound, fraction precision
    g, p, lb, ub, frac_digits = 1, 2, 1, 100, 3
    # Assuming uniform unit traffic.
    traffic_vol = [1] * g

    runtimes = []
    for _ in range(10):
        runtime = []
        for g in range(1, 8, 1):
            input_groups = input_groups_gen(2**g, p, lb, ub, frac_digits)
            start = time.time_ns()
            group_reduction = GroupReduction(input_groups, traffic_vol,
                                             table_limit)
            #output_groups = group_reduction.solve_ssmg('L1NORM')
            output_groups = group_reduction.table_fitting_ssmg()
            end = time.time_ns()
            print('Input: %s' % input_groups)
            print('Output: %s' % output_groups)
            '''
            for i in range(len(input_groups)):
                diff = l1_norm_diff(input_groups[i], output_groups[i])
                print('Group {} L1 Norm: {}'.format(i, diff))
            '''
            print('Solving time (msec):', (end - start)/10**6)
            runtime.append((end - start)/10**6)
        runtimes.append(runtime)
    print('runtime', runtimes)

    with open('sweep.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([str(2**g) + ' groups' for g in range(8)])
        writer.writerows(runtimes)
