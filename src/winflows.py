#!/usr/bin/env python3
"""Windowed flow export, for captures whose connections stay open for a long time.

cicflowmeter turns one long-lived TCP connection into one (or a few) giant flows, which leaves
too few flows to train on. This script cuts the capture into fixed time windows (default 2 s),
exports flows for each window, and concatenates them. Each resulting flow is one connection
seen during one window.

It also adds a "flow id" column = connection + coarse time block (default 30 s). evaluate.py /
sweep.py / groups.py use that column to keep every window of one connection-block on the same
side of the train/test split, so near-identical neighbouring windows never leak across it.
The column is dropped before training, and no time column is added as a feature.

Usage:  python3 winflows.py in.pcap out.csv [--window 2] [--block 30]
(needs `cicflowmeter` on PATH)
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from scapy.all import IP, PcapReader, PcapWriter

DROP_ID = {"flow id", "flow_id"}


def _conn(df):
    cols = {c.lower(): c for c in df.columns}
    si, di = cols.get("src_ip", cols.get("src ip")), cols.get("dst_ip", cols.get("dst ip"))
    sp, dp = cols.get("src_port", cols.get("src port")), cols.get("dst_port", cols.get("dst port"))
    a = df[si].astype(str) + ":" + df[sp].astype(str)
    b = df[di].astype(str) + ":" + df[dp].astype(str)
    return a.where(a < b, b) + "-" + b.where(a < b, a)


def windowed_export(pcap, csv, window=2.0, block=30.0, workers=None):
    exe = shutil.which("cicflowmeter")
    if not exe:
        sys.exit("cicflowmeter not found on PATH; activate the venv where you installed it.")
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    tmp = tempfile.mkdtemp(prefix="winflows_")
    writers, counts = {}, {}
    try:
        for p in PcapReader(pcap):
            if not p.haslayer(IP):
                continue
            w = int(float(p.time) // window)
            if w not in writers:
                writers[w] = PcapWriter(os.path.join(tmp, f"w{w}.pcap"), sync=False)
                counts[w] = 0
            writers[w].write(p)
            counts[w] += 1
        for wr in writers.values():
            wr.close()

        def run(w):
            src, dst = os.path.join(tmp, f"w{w}.pcap"), os.path.join(tmp, f"w{w}.csv")
            subprocess.run([exe, "-f", src, "-c", dst], capture_output=True, text=True)
            if not os.path.exists(dst) or os.path.getsize(dst) == 0:
                return None
            df = pd.read_csv(dst)
            if df.empty:
                return None
            df = df.drop(columns=[c for c in df.columns if c.strip().lower() in DROP_ID])
            df["flow id"] = _conn(df) + "|" + str(int((w * window) // block))
            return df

        with ThreadPoolExecutor(max_workers=workers) as ex:
            parts = [d for d in ex.map(run, sorted(writers)) if d is not None]
        if not parts:
            sys.exit(f"windowed export produced no flows for {pcap}")
        out = pd.concat(parts, ignore_index=True)
        out.to_csv(csv, index=False)
        print(f"  {os.path.basename(pcap)}: {len(writers)} windows -> {len(out)} flows", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--window", type=float, default=2.0, help="window length in seconds")
    ap.add_argument("--block", type=float, default=30.0, help="group block length in seconds")
    a = ap.parse_args()
    windowed_export(a.input, a.output, a.window, a.block)
