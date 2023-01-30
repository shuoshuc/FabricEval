#!/usr/bin/env python
# -*- coding: utf-8 -*-
# A simple script that parses the FabricSim log and extracts group weight ratrio
# into a csv. To use it, execute:
# $ python3 parse.py <path-to-logfile> <path-to-output-csv>

import csv
import re
import sys

if __name__ == "__main__":
    logfile, csvfile = sys.argv[1], sys.argv[2]
    rows = []
    p = re.compile("toy3-c(.*)-ab1.*orig max ratio (.*)")
    with open(logfile, 'r') as f:
        for line in f:
            result = p.search(line)
            if result:
                rows.append([int(result.group(1)), float(result.group(2))])

    rows.sort(key=lambda x: x[0])
    print(f'avg ratio gen1: {sum(n for _, n in rows[:22*8-1]) / 22}')
    print(f'avg ratio gen2: {sum(n for _, n in rows[22*8-1:44*8]) / 22}')
    print(f'avg ratio gen3: {sum(n for _, n in rows[44*8:]) / 21}')
    print(f'max ratio gen1: {max(n for _, n in rows[:22*8-1])}')
    print(f'max ratio gen2: {max(n for _, n in rows[22*8-1:44*8])}')
    print(f'max ratio gen3: {max(n for _, n in rows[44*8:])}')
    with open(csvfile, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['cluster', 'ratio'])
        for row in rows:
            writer.writerow(row)
