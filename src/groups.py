#!/usr/bin/env python3
"""Robust-feature test: train detectors with whole feature groups removed.

Jitter changes timing features and padding changes size features. This script trains the same
three models with
  baseline        all features
  no-size         packet / byte size features removed          (should resist padding)
  no-time         duration / inter-arrival / rate features removed  (should resist jitter)
  no-size+time    both removed (only counts, flags, window sizes remain)
  adv@S           adversarial training at strength S (for comparison)
and reports clean F1, false-positive rate, model size, and the detection rate (DR) under
jitter, padding and both at every strength. The point is the trade-off: how much clean
detection is lost for the robustness gained.

It reuses the flow CSVs that sweep.py cached in --workdir (it never re-exports unless a file is
missing), so run sweep.py first with the same --pcaps / --strengths / --workdir.

Usage:
  python3 groups.py --pcaps mix.pcap mix2.pcap mix3.pcap --attacker 127.0.0.99 \
      --workdir sweepwork --strengths 0.25,0.5,0.75,1 --adv-strengths 1 --seeds 5
"""
import argparse
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from defense import make_models
from evaluate import features, flow_key
from sweep import MODES, load, parse_list, prepare

# feature groups, matched on the lower-case column name (cicflowmeter column names)
GROUPS = {
    "size": re.compile(r"pkt_len|totlen|seg_size|pkt_size|(?<!win_)byts|header_len"),
    "time": re.compile(r"iat|duration|_s$|active_|idle_"),
}


def keep_cols(cols, drop_groups):
    return [c for c in cols if not any(GROUPS[g].search(c.lower()) for g in drop_groups)]


def run_split(ben, mal, pert, seed, ratio, regimes, tag):
    ben_s = ben.sample(n=min(len(ben), ratio * len(mal)), random_state=seed)
    data = pd.concat([ben_s.assign(_y=0), mal.assign(_y=1)], ignore_index=True)
    keys = flow_key(data).values
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(data, groups=keys))
    y = data["_y"].values
    X = features(data.drop(columns=["_y"]))
    cols = list(X.columns)
    tr_keys, te_keys = set(keys[tr]), set(keys[te])
    xp_tr, xp_te = {}, {}
    for n, p in pert.items():
        pk = flow_key(p)
        xp = features(p, cols)
        xp_tr[n] = xp[pk.isin(tr_keys).values]
        xp_te[n] = xp[pk.isin(te_keys).values]
    rows = []
    for rname, (names, drop) in regimes.items():
        rc = keep_cols(cols, drop)
        Xtr, ytr = X.iloc[tr][rc], y[tr]
        if names:
            extra = [xp_tr[n][rc] for n in names if len(xp_tr[n])]
            Xtr = pd.concat([Xtr] + extra, ignore_index=True)
            ytr = np.concatenate([ytr] + [np.ones(len(e), dtype=int) for e in extra])
        for mname, m in make_models(seed).items():
            m.fit(Xtr, ytr)
            pred = m.predict(X.iloc[te][rc])
            yt = y[te]
            row = {"capture": tag, "seed": seed, "regime": rname, "model": mname,
                   "features": len(rc),
                   "F1": f1_score(yt, pred, zero_division=0),
                   "FPR": float(pred[yt == 0].mean()) if (yt == 0).any() else np.nan,
                   "size_KB": len(pickle.dumps(m)) / 1024}
            for n in pert:
                row[n] = float(m.predict(xp_te[n][rc]).mean()) if len(xp_te[n]) else np.nan
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcaps", nargs="+", required=True)
    ap.add_argument("--attacker", required=True)
    ap.add_argument("--workdir", default="sweepwork")
    ap.add_argument("--strengths", default="0.25,0.5,0.75,1")
    ap.add_argument("--adv-strengths", default="1", help="strengths for the adversarial-training comparison")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ratio", type=int, default=20)
    a = ap.parse_args()

    strengths, adv_strengths = parse_list(a.strengths), parse_list(a.adv_strengths)
    bad = [s for s in adv_strengths if s not in strengths]
    if bad:
        sys.exit(f"--adv-strengths {bad} must also appear in --strengths")
    os.makedirs(a.workdir, exist_ok=True)

    all_rows, shown = [], False
    for i, pcap in enumerate(a.pcaps):
        print(f"[{i + 1}/{len(a.pcaps)}] {pcap}", flush=True)
        clean_csv, pert_csv = prepare(pcap, a, strengths)
        ben, mal, pert = load(clean_csv, pert_csv, a.attacker)
        if not shown:
            cols = list(features(pd.concat([ben, mal]).head(5)).columns)
            for g in GROUPS:
                dropped = [c for c in cols if GROUPS[g].search(c.lower())]
                print(f"  group '{g}' removes {len(dropped)} of {len(cols)} features: {', '.join(dropped)}")
            shown = True
        regimes = {"baseline": ([], ()), "no-size": ([], ("size",)), "no-time": ([], ("time",)),
                   "no-size+time": ([], ("size", "time"))}
        for s in adv_strengths:
            regimes[f"adv@{s:g}"] = ([n for n in pert if n.endswith(f"@{s:g}")], ())
        for seed in range(a.seeds):
            all_rows += run_split(ben, mal, pert, seed, a.ratio, regimes, os.path.basename(pcap))

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(a.workdir, "groups_results.csv"), index=False)
    reps = df.groupby(["capture", "seed"]).ngroups
    print(f"\n{reps} repetitions ({len(a.pcaps)} captures x {a.seeds} splits); cells are mean±std")
    idx = pd.MultiIndex.from_frame(df[["regime", "model"]].drop_duplicates())

    def cell(col):
        g = df.groupby(["regime", "model"], sort=False)[col]
        mean, std = g.mean(), g.std().fillna(0)
        return [f"{mean[k]:.2f}±{std[k]:.2f}" for k in idx]

    g = df.groupby(["regime", "model"], sort=False)
    clean = pd.DataFrame({"features": [int(round(g["features"].mean()[k])) for k in idx],
                          "F1": cell("F1"), "FPR": cell("FPR"),
                          "size_KB": [round(g["size_KB"].mean()[k], 1) for k in idx]}, index=idx)
    print("\nclean test traffic\n")
    print(clean.reset_index().to_string(index=False))
    for mname in MODES:
        t = pd.DataFrame({f"strength {s:g}": cell(f"{mname}@{s:g}") for s in strengths}, index=idx)
        print(f"\ndetection rate under {mname}\n")
        print(t.reset_index().to_string(index=False))
    print(f"\nraw rows saved to {os.path.join(a.workdir, 'groups_results.csv')}")


if __name__ == "__main__":
    main()
