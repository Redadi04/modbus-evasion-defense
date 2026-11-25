#!/usr/bin/env python3
"""CIC Modbus 2023, compromised-IED scenario: labelled flows and baseline detectors.

Steps (each cached in --workdir):
  1. crop the pcap to a UTC time range (classic pcap, no editcap needed);
  2. export flows with cicflowmeter (run with TZ=UTC so flow timestamps are UTC);
  3. label a flow malicious if it involves the attacker host and overlaps an attack window from the
     attack logs (windows = first to last log line of one TransactionID, padded);
  4. train DT / RF / HGB (benign:malicious capped at --ratio:1, 80/20 split) and print F1,
     precision, recall on the clean test split, to compare with Table 1 of Galli et al.

Usage (venv active, cicflowmeter on PATH):
  TZ=UTC python3 cicmodbus.py --pcap <ied1b-0.pcap> --logs "<attack logs dir>" --workdir ~/modbus/cicwork \
      --start "2023-03-23 12:00:00" --end "2023-03-23 22:00:00"
"""
import argparse
import glob
import os
import shutil
import struct
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from defense import make_models
from evaluate import features, find, ip_cols, keep_modbus, norm, port_cols


def crop(pcap, out, t0, t1):
    """Copy the packets with t0 <= time < t1 (epoch seconds) from a classic pcap."""
    if os.path.exists(out) and os.path.getsize(out) > 24:
        return
    n = kept = 0
    with open(pcap, "rb") as f, open(out, "wb") as g:
        head = f.read(24)
        magic = head[:4]
        if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
            end = "<"
        elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
            end = ">"
        else:
            sys.exit("not a classic pcap file (pcapng?); convert it first with tcpdump -r in -w out.pcap")
        nano = magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
        g.write(head)
        while True:
            rh = f.read(16)
            if len(rh) < 16:
                break
            sec, frac, incl, _ = struct.unpack(end + "IIII", rh)
            data = f.read(incl)
            if len(data) < incl:
                break
            n += 1
            t = sec + frac / (1e9 if nano else 1e6)
            if t0 <= t < t1:
                g.write(rh)
                g.write(data)
                kept += 1
            elif t >= t1:
                break
    print(f"cropped {kept} of {n} packets read -> {out}")


def export_flows(pcap, csv):
    if os.path.exists(csv) and os.path.getsize(csv) > 0:
        return
    win = float(os.environ.get("WIN", "0"))
    if win > 0:                                   # 2 s (or WIN s) windows, see winflows.py
        from winflows import windowed_export
        windowed_export(pcap, csv, win, 30.0)
        return
    exe = shutil.which("cicflowmeter")
    if not exe:
        sys.exit("cicflowmeter not found on PATH; activate the venv.")
    r = subprocess.run([exe, "-f", pcap, "-c", csv], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(csv) or os.path.getsize(csv) == 0:
        sys.exit(f"flow export failed:\n{(r.stderr or r.stdout)[-800:]}")


def secs(col):
    """UTC epoch seconds, independent of the pandas time resolution."""
    return (pd.to_datetime(col, utc=True) - pd.Timestamp(0, tz="UTC")).dt.total_seconds()


def attack_windows(logs_dir, pad_single):
    """[(start, end, attack name)] in epoch seconds (log times are UTC)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(logs_dir, "**", "*.csv"), recursive=True)):
        d = pd.read_csv(p)
        d.columns = [c.strip() for c in d.columns]
        d["file"] = p
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d = d[~d["Attack"].astype(str).str.strip().str.lower().str.startswith("baseline")].copy()
    d["t"] = secs(d["Timestamp"])
    out = []
    for (_, _), g in d.groupby(["file", "TransactionID"]):
        g = g.sort_values("t")
        name = str(g["Attack"].iloc[0]).strip().split(":")[0].replace("Starting ", "").strip(" .")
        t0, t1 = float(g["t"].iloc[0]), float(g["t"].iloc[-1])
        single = len(g) == 1
        out.append((t0, t1 + (pad_single if single else 0.0), name, single))
    return pd.DataFrame(out, columns=["start", "end", "name", "single"]).sort_values("start").reset_index(drop=True)


def label(flows, win, attacker, pad, dur_unit):
    si, di = ip_cols(flows)
    ts = find(flows, "timestamp")
    du = find(flows, "flow_duration", "flow duration")
    t = secs(flows[ts])
    dur = pd.to_numeric(flows[du], errors="coerce").fillna(0.0) / (1e6 if dur_unit == "us" else 1.0)
    f0, f1 = t.values, (t + dur).values
    involved = ((flows[si].astype(str) == attacker) | (flows[di].astype(str) == attacker)).values
    y = np.zeros(len(flows), dtype=int)
    kind = np.array([""] * len(flows), dtype=object)
    for r in win.itertuples():
        hit = involved & (f1 >= r.start - pad) & (f0 <= r.end + pad)
        new = hit & (y == 0)
        kind[new] = r.name
        y[hit] = 1
    return y, kind, dur


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--logs", required=True, help="the 'attack logs' folder of the scenario")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--start", required=True, help="UTC, YYYY-MM-DD HH:MM:SS")
    ap.add_argument("--end", required=True)
    ap.add_argument("--attacker", default="185.175.0.5")
    ap.add_argument("--pad", type=float, default=2.0, help="seconds added each side of an attack window")
    ap.add_argument("--pad-single", type=float, default=5.0, help="length given to one-line log events")
    ap.add_argument("--duration-unit", choices=["s", "us"], default="s")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ratio", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)

    t0 = pd.Timestamp(a.start, tz="UTC").timestamp()
    t1 = pd.Timestamp(a.end, tz="UTC").timestamp()
    cp = os.path.join(a.workdir, "crop.pcap")
    crop(a.pcap, cp, t0, t1)
    csv = os.path.join(a.workdir, "crop_flows.csv")
    export_flows(cp, csv)

    flows = keep_modbus(norm(pd.read_csv(csv)), "crop")
    win = attack_windows(a.logs, a.pad_single)
    win = win[(win["end"] >= t0) & (win["start"] <= t1)]
    print(f"\nattack events inside the cropped range: {len(win)} "
          f"({int(win['single'].sum())} one-line events given {a.pad_single:.0f} s)")
    print(win.groupby("name").size().to_string())

    y, kind, dur = label(flows, win, a.attacker, a.pad, a.duration_unit)
    flows = flows.reset_index(drop=True)
    print(f"\nflows: {len(flows)} total, {int(y.sum())} malicious, {int((y == 0).sum())} benign; "
          f"flow duration median {np.median(dur):.3f} max {dur.max():.1f} (unit assumed {a.duration_unit})")
    print(pd.Series(kind[y == 1]).value_counts().to_string())
    flows.assign(_y=y, _kind=kind).to_csv(os.path.join(a.workdir, "labelled_flows.csv"), index=False)
    if y.sum() < 20:
        sys.exit("too few malicious flows: widen --start/--end or check --duration-unit")

    X = features(flows)
    rows = []
    for seed in range(a.seeds):
        mal = np.where(y == 1)[0]
        ben_all = np.where(y == 0)[0]
        rng = np.random.RandomState(seed)
        ben = rng.choice(ben_all, size=min(len(ben_all), a.ratio * len(mal)), replace=False)
        idx = np.concatenate([ben, mal])
        tr, te = train_test_split(idx, test_size=0.2, stratify=y[idx], random_state=seed)
        for name, m in make_models(seed).items():
            m.fit(X.iloc[tr], y[tr])
            p = m.predict(X.iloc[te])
            rows.append({"model": name, "F1": f1_score(y[te], p), "precision": precision_score(y[te], p, zero_division=0),
                         "recall": recall_score(y[te], p)})
    df = pd.DataFrame(rows)
    print(f"\nclean test split, {a.seeds} seeds (benign capped at {a.ratio}:1)")
    print(df.groupby("model")[["F1", "precision", "recall"]].agg(["mean", "std"]).round(3).to_string())
    print("\nGalli et al. Table 1, internal IED: DT F1 0.906, RF 0.934, HGB 0.937")


if __name__ == "__main__":
    main()
