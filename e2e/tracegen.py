import csv
import random
import re
import sys
from pathlib import Path

import numpy as np

import common.flags as FLAG
from traffic.tmgen import tmgen
from traffic.traffic import loadTraffic

NETWORK = 'toy3'
TM_PATH = f'tests/data/{NETWORK}_traffic_gravity.textproto'

MSFT_WEBSEARCH = 'MsftWebSearch.txt'
ALI_STORAGE = 'AliStorage.txt'
GOOGLE_RPC = 'GoogleRPC.txt'
FB_HADOOP = 'FbHadoop.txt'

def loadCDF(filename):
    '''
    Loads the given csv file and parses it into a CDF.
    '''
    cdf = []
    with open(filename, 'r', encoding='utf-8') as f:
        # saves the data points as [[x_i, p_i] ...]
        for line in f.readlines():
            x, p = map(float, line.strip().split(' '))
            cdf.append([x, p])
    return cdf

class CustomRand:
    '''
    A custom random variable class that fits the given CDF. It is able to
    generate data points following the given CDF.
    '''
    def testCdf(self, cdf):
        if cdf[0][1] != 0:
            return False
        if cdf[-1][1] != 100:
            return False
        for i in range(1, len(cdf)):
            if cdf[i][1] <= cdf[i-1][1] or cdf[i][0] <= cdf[i-1][0]:
                return False
        return True

    def setCdf(self, cdf):
        if not self.testCdf(cdf):
            return False
        self.cdf = cdf
        return True

    def getAvg(self):
        s = 0
        last_x, last_y = self.cdf[0]
        for c in self.cdf[1:]:
            x, y = c
            s += (x + last_x)/2.0 * (y - last_y)
            last_x = x
            last_y = y
        return s/100

    def rand(self):
        r = random.random() * 100
        return self.getValueFromPercentile(r)

    def getPercentileFromValue(self, x):
        if x < 0 or x > self.cdf[-1][0]:
            return -1
        for i in range(1, len(self.cdf)):
            if x <= self.cdf[i][0]:
                x0, y0 = self.cdf[i-1]
                x1, y1 = self.cdf[i]
                return y0 + (y1-y0)/(x1-x0)*(x-x0)

    def getValueFromPercentile(self, y):
        for i in range(1, len(self.cdf)):
            if y <= self.cdf[i][1]:
                x0,y0 = self.cdf[i-1]
                x1,y1 = self.cdf[i]
                return x0 + (x1-x0)/(y1-y0)*(y-y0)

    def getIntegralY(self, y):
        s = 0
        for i in range(1, len(self.cdf)):
            x0, y0 = self.cdf[i-1]
            x1, y1 = self.cdf[i]
            if y <= self.cdf[i][1]:
                s += 0.5 * (x0 + x0+(x1-x0)/(y1-y0)*(y-y0))*(y-y0) / 100.
                break
            else:
                s += 0.5 * (x1 + x0) * (y1 - y0) / 100.
        return s

def tracegen(TM, cluster_vector, rv, duration, load):
    '''
    Generates workload traces that matches the input demand traffic matrix while
    conforming to the flow size distribution defined in `rv`.

    TM: traffic matrix of format
        [[src node, src cluster id, dst node, dst cluster id, demand], ...]
    cluster_vector: a vector of cluster speed ratio, based speed is 40Gbps.
    rv: random variable that models the workload flow size distribution.
    duration: time duration (in nsec) the TM is measured on.
    load: link load between 0 and 1.

    Returns a trace of format [src, dst, flow size (Bytes), start time (nsec)].
    '''
    # Base speed is 40Gbps in bps.
    BASE_BW = 40 * 1000 * 1000 * 1000

    trace = []
    # Start time of the last flow in the entire trace.
    t_last_flow = 0
    for src, sidx, dst, didx, demand in TM:
        # Speed auto-negotiation.
        BW = BASE_BW * min(cluster_vector[int(sidx) - 1],
                           cluster_vector[int(didx) - 1])
        # Get target flow size between two nodes. demand is in Mbps, duration
        # is in nsec. target_size is in bytes.
        target_size = (int(demand) / 8. * 1000000) * (duration / 1000000000)

        tot_size = 0
        prev_time = 0
        avg_inter_arrival_nsec = 1 / (BW * load / 8. / rv.getAvg()) * 1000000000
        while tot_size < target_size:
            flow_size = int(rv.rand())
            iat_ns = int(np.random.exponential(avg_inter_arrival_nsec))
            prev_time += iat_ns
            trace.append([src, sidx, dst, didx, flow_size, prev_time])
            tot_size += flow_size
        if prev_time > duration:
            print(f'[WARN] trace {src} => {dst}: {target_size} exceeds '
                  f'duration: {prev_time} > {duration} nsec.')
        t_last_flow = max(t_last_flow, prev_time)
    print(f'Last flow start time {t_last_flow} nsec.')
    return trace

if __name__ == "__main__":
    # Selected workload type.
    WORKLOAD = MSFT_WEBSEARCH
    # Trace duration 50 msec.
    DURATION = 20 * 1000 * 1000
    # Link load 40%.
    LOAD = 0.4

    # Loads all workload CDFs.
    CDF = {
        MSFT_WEBSEARCH: None,
        ALI_STORAGE: None,
        GOOGLE_RPC: None,
        FB_HADOOP: None
    }
    for workload in CDF.keys():
        CDF[workload] = CustomRand()
        if not CDF[workload].setCdf(loadCDF(f'traffic/data/{workload}')):
            print(f"[ERROR] Invalid CDF, workload: {workload}")
            continue

    # Each demand entry looks like:
    # [src node, src cluster id, dst node, dst cluster id, demand (Mbps)].
    rawTM = []
    proto_traffic = loadTraffic(TM_PATH)

    pattern = re.compile("(.*)-c([0-9]+)-ab1-s1i([0-9]+)")
    for demand_entry in proto_traffic.demands:
        src, dst = demand_entry.src, demand_entry.dst
        # Sanity check: src and dst cannot be the same.
        if src == dst:
            print(f'[ERROR] Traffic parsing: src {src} and dst {dst} cannot'
                  f' be the same!')
            break
        # Sanity check: only positive demand allowed.
        vol = demand_entry.volume_mbps
        if vol <= 0:
            print(f'[ERROR] Traffic parsing: encountered negative demand: '
                  f'{vol} on {src} => {dst}.')
            break

        match_src, match_dst = pattern.search(src), pattern.search(dst)
        if not match_src or not match_dst:
            continue
        netname = match_src.group(1)
        src_cid, dst_cid = match_src.group(2), match_dst.group(2)
        src_tid, dst_tid = match_src.group(3), match_dst.group(3)
        rawTM.append([f'{netname}-c{src_cid}-t{src_tid}', f'{src_cid}',
                      f'{netname}-c{dst_cid}-t{dst_tid}', f'{dst_cid}', vol])

    # Generates trace.
    speed_vec = np.array([1]*22 + [2.5]*22 + [5]*21)
    trace = tracegen(rawTM, speed_vec, CDF[WORKLOAD], DURATION, LOAD)

    # Writes trace to filesystem as a csv.
    logpath = Path(sys.argv[1])
    logpath.mkdir(parents=True, exist_ok=True)
    with (logpath / f'{NETWORK}-trace.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerows(trace)
