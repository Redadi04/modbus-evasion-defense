#!/usr/bin/env python3
"""CIC Modbus 2023: benign class from the benign captures, attack class from the attack capture.

Uses labelled_flows.csv written by cicmodbus.py (flows of the attack capture, with _y from the attack
logs) and exports flows from one or more benign pcaps. Two labelling rules for the attack class:
  logs : malicious = attacker flows overlapping a log window (_y == 1)
  host : malicious = every attack-capture flow involving the attacker host
Benign = flows of the benign pcaps (same interface, no attack). Prints clean F1 / precision / recall
for DT, RF, HGB (benign capped at --ratio:1, 80/20 split) to compare with Table 1 of Galli et al.

Usage (venv active): python3 cicbenign.py --attack-flows ~/modbus/cicwork/labelled_flows.csv \
        --benign-pcaps b4.pcap b5.pcap --workdir ~/modbus/cicwork
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from defense import make_models
from evaluate import features, ip_cols, keep_modbus, norm


def export_flows(pcap, csv):
    if os.path.exists(csv) and os.path.getsize(csv) > 0:
        return
    exe = shutil.which("cicflowmeter")
    if not exe:
        sys.exit("cicflowmeter not found on PATH; activate the venv.")
    r = subprocess.run([exe, "-f", pcap, "-c", csv], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(csv) or os.path.getsize(csv) == 0:
        sys.exit(f"flow export failed for {pcap}:\n{(r.stderr or r.stdout)[-800:]}")


def evaluate(mal, ben, seeds, ratio):
    """mal, ben: flow tables. Returns a table of mean F1 / precision / recall per model."""
    data = pd.concat([ben.assign(_y=0), mal.assign(_y=1)], ignore_index=True)
    y = data["_y"].values
    X = features(data.drop(columns=["_y"] + [c for c in data.columns if c.startswith("_kind")]))
    rows = []
    for seed in range(seeds):
        rng = np.random.RandomState(seed)
        m_idx, b_idx = np.where(y == 1)[0], np.where(y == 0)[0]
        b_idx = rng.choice(b_idx, size=min(len(b_idx), ratio * len(m_idx)), replace=False)
        idx = np.concatenate([b_idx, m_idx])
        tr, te = train_test_split(idx, test_size=0.2, stratify=y[idx], random_state=seed)
        for name, m in make_models(seed).items():
            m.fit(X.iloc[tr], y[tr])
            p = m.predict(X.iloc[te])
            rows.append({"model": name, "F1": f1_score(y[te], p), "precision": precision_score(y[te], p, zero_division=0),
                         "recall": recall_score(y[te], p)})
    return pd.DataFrame(rows).groupby("model")[["F1", "precision", "recall"]].agg(["mean", "std"]).round(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack-flows", required=True)
    ap.add_argument("--benign-pcaps", nargs="+", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ratio", type=int, default=20)
    a = ap.parse_args()

    att = pd.read_csv(a.attack_flows)
    parts = []
    for p in a.benign_pcaps:
        csv = os.path.join(a.workdir, os.path.splitext(os.path.basename(p))[0] + "_flows.csv")
        export_flows(p, csv)
        parts.append(keep_modbus(norm(pd.read_csv(csv)), os.path.basename(p)))
    ben = pd.concat(parts, ignore_index=True)
    print(f"\nbenign flows: {len(ben)}; attack-capture flows: {len(att)} ({int(att['_y'].sum())} in log windows)")

    keep = [c for c in att.columns if c not in ("_y", "_kind")]
    for name, mal in {"logs": att[att["_y"] == 1], "host": att}.items():
        if len(mal) < 20:
            print(f"\nrule {name}: too few malicious flows ({len(mal)})")
            continue
        print(f"\nrule {name}: {len(mal)} malicious vs {len(ben)} benign (cap {a.ratio}:1), {a.seeds} seeds")
        print(evaluate(mal[keep], ben, a.seeds, a.ratio).to_string())
    print("\nGalli et al. Table 1, internal IED: DT F1 0.906, RF 0.934, HGB 0.937")


if __name__ == "__main__":
    main()
