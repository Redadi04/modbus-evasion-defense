#!/usr/bin/env python3
"""Control experiment: does adversarial training help because the copies are PERTURBED, or just
because it adds more malicious training flows?

Regimes (same clean training flows for all):
  baseline   clean training flows only
  dup@K      control: the training attacker flows repeated K times, UNPERTURBED
             (K = number of perturbed copies that adv@S adds, so both add the same number of rows)
  adv@S      adversarial training: perturbed copies (jitter, padding, both) of the training
             attacker flows at strength S

If adv@S beats dup@K under perturbation, the perturbation itself is what helps. If dup@K does
about as well, the gain is just extra weight on the malicious class.

Reports clean F1, recall and false-positive rate, and the detection rate on perturbed test
attacker flows. Reuses the flow CSVs that sweep.py cached in --workdir (run sweep.py first with
the same --pcaps / --strengths / --workdir, plus --window if you used it there).

Usage:
  python3 control.py --pcaps testbed/m1.pcap testbed/m2.pcap testbed/m3.pcap --attacker 127.0.0.99 \
      --workdir mimicwork --strengths 0.5,1 --adv-strengths 1 --seeds 5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

import sweep
from defense import make_models
from evaluate import features, flow_key
from sweep import MODES, load, parse_list, prepare


def run_split(ben, mal, pert, seed, ratio, adv_strengths, tag):
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

    Xtr, ytr = X.iloc[tr], y[tr]
    mal_tr = Xtr[ytr == 1]
    regimes = {"baseline": (Xtr, ytr)}
    for s in adv_strengths:
        names = [n for n in pert if n.endswith(f"@{s:g}")]
        extra = [xp_tr[n] for n in names if len(xp_tr[n])]
        k = len(extra)
        # control: the same number of extra malicious rows, but unperturbed copies of the training attacker flows
        dup = pd.concat([Xtr] + [mal_tr] * k, ignore_index=True)
        ydup = np.concatenate([ytr] + [np.ones(len(mal_tr), dtype=int)] * k)
        regimes[f"dup@{k}"] = (dup, ydup)
        adv = pd.concat([Xtr] + extra, ignore_index=True)
        yadv = np.concatenate([ytr] + [np.ones(len(e), dtype=int) for e in extra])
        regimes[f"adv@{s:g}"] = (adv, yadv)

    rows = []
    for rname, (Xr, yr) in regimes.items():
        for mname, m in make_models(seed).items():
            m.fit(Xr, yr)
            pred = m.predict(X.iloc[te])
            yt = y[te]
            row = {"capture": tag, "seed": seed, "regime": rname, "model": mname,
                   "F1": f1_score(yt, pred, zero_division=0),
                   "recall": recall_score(yt, pred, zero_division=0),
                   "FPR": float(pred[yt == 0].mean()) if (yt == 0).any() else np.nan,
                   "train_rows": len(Xr)}
            for n in pert:
                row[n] = float(m.predict(xp_te[n]).mean()) if len(xp_te[n]) else np.nan
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcaps", nargs="+", required=True)
    ap.add_argument("--attacker", required=True)
    ap.add_argument("--workdir", default="sweepwork")
    ap.add_argument("--strengths", default="0.5,1")
    ap.add_argument("--adv-strengths", default="1")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ratio", type=int, default=20)
    ap.add_argument("--window", type=float, default=0.0, help="use the same value as in sweep.py (0 = off)")
    ap.add_argument("--block", type=float, default=30.0)
    a = ap.parse_args()
    sweep.WIN["window"], sweep.WIN["block"] = a.window, a.block

    strengths, adv_strengths = parse_list(a.strengths), parse_list(a.adv_strengths)
    bad = [s for s in adv_strengths if s not in strengths]
    if bad:
        sys.exit(f"--adv-strengths {bad} must also appear in --strengths")
    os.makedirs(a.workdir, exist_ok=True)

    all_rows = []
    for i, pcap in enumerate(a.pcaps):
        print(f"[{i + 1}/{len(a.pcaps)}] {pcap}", flush=True)
        clean_csv, pert_csv = prepare(pcap, a, strengths)
        ben, mal, pert = load(clean_csv, pert_csv, a.attacker)
        for seed in range(a.seeds):
            all_rows += run_split(ben, mal, pert, seed, a.ratio, adv_strengths, os.path.basename(pcap))

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(a.workdir, "control_results.csv"), index=False)
    reps = df.groupby(["capture", "seed"]).ngroups
    print(f"\n{reps} repetitions ({len(a.pcaps)} captures x {a.seeds} splits); cells are mean±std")
    idx = pd.MultiIndex.from_frame(df[["regime", "model"]].drop_duplicates())

    def cell(col):
        g = df.groupby(["regime", "model"], sort=False)[col]
        mean, std = g.mean(), g.std().fillna(0)
        return [f"{mean[k]:.2f}±{std[k]:.2f}" for k in idx]

    g = df.groupby(["regime", "model"], sort=False)
    clean = pd.DataFrame({"train_rows": [int(round(g["train_rows"].mean()[k])) for k in idx],
                          "F1": cell("F1"), "recall": cell("recall"), "FPR": cell("FPR")}, index=idx)
    print("\nclean test traffic\n")
    print(clean.reset_index().to_string(index=False))
    for mname in MODES:
        t = pd.DataFrame({f"strength {s:g}": cell(f"{mname}@{s:g}") for s in strengths}, index=idx)
        print(f"\ndetection rate under {mname}\n")
        print(t.reset_index().to_string(index=False))
    print(f"\nraw rows saved to {os.path.join(a.workdir, 'control_results.csv')}")


if __name__ == "__main__":
    main()
