#!/usr/bin/env python3
"""Train Decision Tree / Random Forest / Histogram Gradient Boosting on flow records
and report clean-traffic scores plus detection rate on perturbed malicious flows.

Follows Galli et al. (2026) Sections 5.1-5.2 and 6, written from the paper's description.

Two ways to give it data:

 A) Separate benign and malicious flow CSVs (as in the CIC Modbus 2023 layout):
      python3 evaluate.py --benign benign.csv --malicious mal.csv \
          --perturbed static-jitter=pj.csv static-padding=pp.csv

 B) One flow CSV from a capture that contains both normal and attack traffic, labelled
    by the attacker's IP address (flows to or from that IP are malicious):
      python3 evaluate.py --flows flows.csv --attacker 172.27.224.80 \
          --perturbed static-jitter=flows_jitter.csv static-padding=flows_pad.csv

--perturbed files are flow exports of the perturbed pcap. With --attacker, only the
attacker's flows are read from them. Flow CSVs may come from CICFlowMeter (Java) or the
`cicflowmeter` Python port; both column spellings are handled.
"""
import argparse
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

# identifier / bookkeeping columns that must never be used as features
DROP = ["flow id", "flow_id", "src ip", "source ip", "src_ip", "dst ip", "destination ip", "dst_ip",
        "src port", "source port", "src_port", "dst port", "destination port", "dst_port",
        "timestamp", "protocol", "label"]


def norm(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def find(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


def ip_cols(df):
    return find(df, "src ip", "source ip", "src_ip"), find(df, "dst ip", "destination ip", "dst_ip")


def port_cols(df):
    return find(df, "src port", "source port", "src_port"), find(df, "dst port", "destination port", "dst_port")


def keep_modbus(df, name):
    """Keep Modbus TCP only (port 502)."""
    sp, dp = port_cols(df)
    if not (sp and dp):
        return df
    keep = (pd.to_numeric(df[sp], errors="coerce") == 502) | (pd.to_numeric(df[dp], errors="coerce") == 502)
    print(f"{name}: kept {int(keep.sum())}/{len(df)} flows on port 502")
    return df[keep].copy()


def attacker_mask(df, attacker):
    si, di = ip_cols(df)
    if not (si and di):
        raise SystemExit("No source/destination IP columns found in the flow CSV; cannot split by attacker.")
    return (df[si].astype(str) == attacker) | (df[di].astype(str) == attacker)


def flow_key(df):
    """Connection identity, the same before and after perturbation (perturbation keeps ports)."""
    fid = find(df, "flow id", "flow_id")
    if fid:
        return df[fid].astype(str)
    si, di = ip_cols(df)
    sp, dp = port_cols(df)
    if si and di and sp and dp:
        a = df[si].astype(str) + ":" + df[sp].astype(str)
        b = df[di].astype(str) + ":" + df[dp].astype(str)
        return pd.Series(np.where(a < b, a + "-" + b, b + "-" + a), index=df.index)
    return pd.Series(np.arange(len(df)).astype(str), index=df.index)


def features(df, cols=None):
    drop = [c for c in df.columns if c.lower() in DROP]
    x = df.drop(columns=drop).apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if cols is not None:
        x = x.reindex(columns=cols, fill_value=0.0)
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benign", help="flow CSV with benign flows only")
    ap.add_argument("--malicious", help="flow CSV with malicious flows only")
    ap.add_argument("--flows", help="one flow CSV with both kinds; needs --attacker")
    ap.add_argument("--attacker", help="attacker IP; flows to/from it are malicious")
    ap.add_argument("--perturbed", nargs="*", default=[], help="name=path.csv pairs")
    ap.add_argument("--ratio", type=int, default=20, help="max benign:malicious ratio (paper: 20)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.flows:
        if not a.attacker:
            ap.error("--flows needs --attacker")
        allf = keep_modbus(norm(pd.read_csv(a.flows)), "flows")
        is_att = attacker_mask(allf, a.attacker)
        ben, mal = allf[~is_att].copy(), allf[is_att].copy()
    elif a.benign and a.malicious:
        ben = keep_modbus(norm(pd.read_csv(a.benign)), "benign")
        mal = keep_modbus(norm(pd.read_csv(a.malicious)), "malicious")
    else:
        ap.error("give either --flows and --attacker, or --benign and --malicious")

    n_ben = min(len(ben), a.ratio * len(mal))
    ben = ben.sample(n=n_ben, random_state=a.seed)
    print(f"benign:malicious = {len(ben)}:{len(mal)}")
    if len(ben) == 0 or len(mal) == 0:
        raise SystemExit("Need at least one benign and one malicious flow.")

    data = pd.concat([ben.assign(_y=0), mal.assign(_y=1)], ignore_index=True)
    groups = flow_key(data).values
    # group split on connection identity so perturbed copies of a training flow never reach the test side
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=a.seed).split(data, groups=groups))
    y = data["_y"].values
    X = features(data.drop(columns=["_y"]))
    cols = list(X.columns)
    test_ids = set(groups[te])
    print(f"train: {len(tr)} flows ({int(y[tr].sum())} malicious); test: {len(te)} flows ({int(y[te].sum())} malicious)")

    pert = {}
    for item in a.perturbed:
        name, path = item.split("=", 1)
        p = keep_modbus(norm(pd.read_csv(path)), f"perturbed[{name}]")
        if a.attacker:
            p = p[attacker_mask(p, a.attacker)]
        p = p[flow_key(p).isin(test_ids)]       # only flows whose connection is test-side
        pert[name] = features(p, cols)
        print(f"perturbed[{name}]: {len(pert[name])} malicious flows usable on the test side")

    models = {
        "DecisionTree": DecisionTreeClassifier(criterion="gini", random_state=a.seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=a.seed, n_jobs=-1),
        "HistGradBoost": HistGradientBoostingClassifier(learning_rate=0.1, max_iter=100, random_state=a.seed),
    }
    rows = []
    for name, m in models.items():
        m.fit(X.iloc[tr], y[tr])
        t0 = time.perf_counter()
        pred = m.predict(X.iloc[te])
        us = (time.perf_counter() - t0) / max(len(te), 1) * 1e6
        row = {"model": name,
               "F1": f1_score(y[te], pred, zero_division=0),
               "Precision": precision_score(y[te], pred, zero_division=0),
               "Recall": recall_score(y[te], pred, zero_division=0),
               "size_KB": len(pickle.dumps(m)) / 1024,
               "us_per_flow": us}
        for pn, px in pert.items():
            row[f"DR[{pn}]"] = float(m.predict(px).mean()) if len(px) else float("nan")
        rows.append(row)
    print()
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
