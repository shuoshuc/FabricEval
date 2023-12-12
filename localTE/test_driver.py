import csv
import sys
import time

from group_reduction import GroupReduction, input_groups_gen

import common.flags as FLAG

if __name__ == "__main__":
    num_runs = 20
    # lower bound, upper bound, fraction precision, table size
    lb, ub, frac_digits, C = 0, 200000, 6, 16*1024

    with open(f'{sys.argv[1]}/grspeed.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'algo', 'num_groups', 'num_ports',
                         'solve_time (msec)'])

        # Single group, sweep num ports. Repeat 20 times for each run.
        print(f'[Single group port sweep]')
        for run in range(num_runs):
            print(f'===== Run {run} starts =====')
            for p in [16, 32, 64]:
                runtime = []
                orig_groups = input_groups_gen(1, p, lb, ub, frac_digits)
                print(f'Input: {orig_groups}')

                for algo in ['eurosys', 'igr', 'dmir']:
                    # Initializes global flags before running the pipeline.
                    if algo == 'eurosys':
                        FLAG.EUROSYS_MOD = False
                    elif algo == 'igr':
                        FLAG.IMPROVED_HEURISTIC = True
                    elif algo == 'dmir':
                        FLAG.IMPROVED_HEURISTIC = True
                    start = time.time_ns()
                    group_reduction = GroupReduction(orig_groups, 1, C)
                    reduced_groups = group_reduction.solve(algo)
                    end = time.time_ns()
                    print(f'Output [{algo}]: {reduced_groups}')
                    solving_time = (end - start)/10**6
                    print('Solving time (msec):', solving_time)
                    writer.writerow([run, algo, 1, p, solving_time])
                    f.flush()
            print(f'===== Run {run} ends =====')

        # Fixed 64 ports, sweep num groups. Repeat 20 times for each run.
        print(f'[Fixed port group sweep]')
        for run in range(num_runs):
            print(f'===== Run {run} starts =====')
            for g in [16, 32, 64]:
                runtime = []
                orig_groups = input_groups_gen(g, 64, lb, ub, frac_digits)
                print(f'Input: {orig_groups}')

                for algo in ['eurosys', 'igr', 'dmir']:
                    # Initializes global flags before running the pipeline.
                    if algo == 'eurosys':
                        FLAG.EUROSYS_MOD = False
                    elif algo == 'igr':
                        FLAG.IMPROVED_HEURISTIC = True
                    elif algo == 'dmir':
                        FLAG.IMPROVED_HEURISTIC = True
                    start = time.time_ns()
                    group_reduction = GroupReduction(orig_groups, 1, C)
                    reduced_groups = group_reduction.solve(algo)
                    end = time.time_ns()
                    print(f'Output [{algo}]: {reduced_groups}')
                    solving_time = (end - start)/10**6
                    print('Solving time (msec):', solving_time)
                    writer.writerow([run, algo, g, 64, solving_time])
                    f.flush()
            print(f'===== Run {run} ends =====')
