#!/usr/bin/env python
# -*- coding: utf-8 -*-
# A simple script that parses the FabricSim log and extracts group reduction
# time into a csv. To use it, execute:
# $ python3 time_extract.py <path-to-logfile> <path-to-output-csv>

import csv
import re
import sys

if __name__ == "__main__":
    logfile, csvfile = sys.argv[1], sys.argv[2]
    time_pts = []
    p = re.compile(".*\[reduceGroups\].*in (.*) sec")
    with open(logfile, 'r') as f:
        for line in f:
            result = p.search(line)
            if result:
                time_pts.append(result.group(1))

    with open(csvfile, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['time (sec)'])
        for time in time_pts:
            writer.writerow([time])
