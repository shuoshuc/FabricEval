#!/usr/bin/env python
# -*- coding: utf-8 -*-

from group_reduction import *
import csv

if __name__ == "__main__":
    table_limit = 16*1024
    # groups, # port per group, lower bound, upper bound, fraction precision
    g, p, lb, ub, frac_digits = 1, 2, 100, 100000, 3

    with open('sweep.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['input_group', 'run', 'solver', 'solve_time (msec)',
                         'l1_norm', 'num_groups', 'num_ports', 'weight_lb',
                         'weight_ub', 'fraction_digits'])

        for p in [2, 4, 8, 16]:
            runtime = []
            input_groups = input_groups_gen(g, p, lb, ub, frac_digits)
            print('Input: %s' % input_groups)
            group_reduction = GroupReduction(input_groups, table_limit)

            # repeat 20 times for each run.
            for run in range(20):
                for solver in ['mip', 'heuristic']:
                    start = time.time_ns()
                    if solver == 'mip':
                        output_groups = group_reduction.solve_sssg()
                    else:
                        output_groups = group_reduction.table_fitting_sssg()
                    end = time.time_ns()
                    print(f'Output [{solver}]: {output_groups}')
                    l1_norm = l1_norm_diff(input_groups, output_groups)
                    print(f'L1 Norm: {l1_norm}')
                    solving_time = (end - start)/10**6
                    print('Solving time (msec):', solving_time)
                    writer.writerow([",".join(map(str, input_groups[0])), run,
                                     solver, solving_time, l1_norm, g, p, lb,
                                     ub, frac_digits])
