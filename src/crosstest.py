#!/usr/bin/env python3
"""Adaptive-attacker test: train on one attacker, test on a different one.

The defenses (adversarial training, removing size or timing features) are built against jitter
and padding. This script asks what happens when the attacker changes its BEHAVIOUR instead:
detectors are trained on captures with attacker A (e.g. the fast-burst writer) and tested on
captures with attacker B (e.g. the one that imitates normal clients).

Regimes (trained only on the --train-pcaps flows):
  baseline        all features, clean training flows
  adv@S           plus perturbed copies of the training attacker flows at strength S
  no-size         26 size features removed
  no-time         27 timing features removed
  no-size+time    both removed
  no-size+adv@S   size features removed and adversarial training

Reports, on the --test-pcaps flows: recall on the unperturbed attacker flows, false-positive rate
on the normal flows, and detection of the attacker flows after jitter / padding / both.
Both sets of captures need flows cached by sweep.py (--train-work / --test-work).

Usage:
  python3 crosstest.py --train-pcaps testbed/mix.pcap testbed/mix2.pcap testbed/mix3.pcap \
      --test-pcaps testbed/m1.pcap testbed/m2.pcap testbed/m3.pcap --attacker 127.0.0.99 \
      --train-work testbed/sweepwork --test-work mimicwork --strengths 0.5,1 --adv-strengths 1
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

import sweep
from defense import make_models
from evaluate import features
from groups import keep_cols
from sweep import MODES, load, parse_list, prepare


def pooled(pcaps, workdir, attacker, strengths, window, block):
    sweep.WIN["window"], sweep.WIN["block"] = window, block
    os.makedirs(workdir, exist_ok=True)
    ns = SimpleNamespace(workdir=workdir, attacker=attacker)
    bens, mals, perts = [], [], {}
    for p in pcaps:
        clean_csv, pert_csv = prepare(p, ns, strengths)
        ben, mal, pert = load(clean_csv, pert_csv, attacker)
        bens.append(ben)
        mals.append(mal)
        for n, d in pert.items():
            perts.setdefault(n, []).append(d)
    return (pd.concat(bens, ignore_index=True), pd.concat(mals, ignore_index=True),
            {n: pd.concat(v, ignore_index=True) for n, v in perts.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-pcaps", nargs="+", required=True)
    ap.add_argument("--test-pcaps", nargs="+", required=True)
    ap.add_argument("--attacker", required=True)
    ap.add_argument("--train-work", required=True)
    ap.add_argument("--test-work", required=True)
    ap.add_argument("--strengths", default="0.5,1")
    ap.add_argument("--adv-strengths", default="1")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ratio", type=int, default=20)
    ap.add_argument("--window", type=float, default=0.0)
    ap.add_argument("--block", type=float, default=30.0)
    a = ap.parse_args()

    strengths, adv_strengths = parse_list(a.strengths), parse_list(a.adv_strengths)
    bad = [s for s in adv_strengths if s not in strengths]
    if bad:
        sys.exit(f"--adv-strengths {bad} must also appear in --strengths")

    print("loading training captures ...", flush=True)
    tr_ben, tr_mal, tr_pert = pooled(a.train_pcaps, a.train_work, a.attacker, strengths, a.window, a.block)
    print(f"  train: benign flows={len(tr_ben)} attacker flows={len(tr_mal)}")
    print("loading test captures ...", flush=True)
    te_ben, te_mal, te_pert = pooled(a.test_pcaps, a.test_work, a.attacker, strengths, a.window, a.block)
    print(f"  test:  benign flows={len(te_ben)} attacker flows={len(te_mal)}")

    rows = []
    for seed in range(a.seeds):
        ben_s = tr_ben.sample(n=min(len(tr_ben), a.ratio * len(tr_mal)), random_state=seed)
        data = pd.concat([ben_s.assign(_y=0), tr_mal.assign(_y=1)], ignore_index=True)
        y = data["_y"].values
        X = features(data.drop(columns=["_y"]))
        cols = list(X.columns)
        xp_tr = {n: features(p, cols) for n, p in tr_pert.items()}
        Xte_ben, Xte_mal = features(te_ben, cols), features(te_mal, cols)
        Xte_pert = {n: features(p, cols) for n, p in te_pert.items()}

        regimes = {"baseline": ([], ()), "no-size": ([], ("size",)), "no-time": ([], ("time",)),
                   "no-size+time": ([], ("size", "time"))}
        for s in adv_strengths:
            names = [n for n in tr_pert if n.endswith(f"@{s:g}")]
            regimes[f"adv@{s:g}"] = (names, ())
            regimes[f"no-size+adv@{s:g}"] = (names, ("size",))
        for rname, (names, drop) in regimes.items():
            rc = keep_cols(cols, drop)
            Xtr, ytr = X[rc], y
            if names:
                extra = [xp_tr[n][rc] for n in names if len(xp_tr[n])]
                Xtr = pd.concat([Xtr] + extra, ignore_index=True)
                ytr = np.concatenate([ytr] + [np.ones(len(e), dtype=int) for e in extra])
            for mname, m in make_models(seed).items():
                m.fit(Xtr, ytr)
                row = {"seed": seed, "regime": rname, "model": mname, "features": len(rc),
                       "recall": float(m.predict(Xte_mal[rc]).mean()),
                       "FPR": float(m.predict(Xte_ben[rc]).mean())}
                for n, xp in Xte_pert.items():
                    row[n] = float(m.predict(xp[rc]).mean()) if len(xp) else np.nan
                rows.append(row)

    df = pd.DataFrame(rows)
    out = os.path.join(a.test_work, "crosstest_results.csv")
    df.to_csv(out, index=False)
    print(f"\n{a.seeds} training runs (different benign samples and model seeds); cells are mean±std")
    idx = pd.MultiIndex.from_frame(df[["regime", "model"]].drop_duplicates())
    g = df.groupby(["regime", "model"], sort=False)

    def cell(col):
        mean, std = g[col].mean(), g[col].std().fillna(0)
        return [f"{mean[k]:.2f}±{std[k]:.2f}" for k in idx]

    tab = pd.DataFrame({"features": [int(round(g["features"].mean()[k])) for k in idx],
                        "recall (clean attacker)": cell("recall"), "FPR (normal flows)": cell("FPR")}, index=idx)
    print("\ntest attacker, unperturbed\n")
    print(tab.reset_index().to_string(index=False))
    for mname in MODES:
        t = pd.DataFrame({f"strength {s:g}": cell(f"{mname}@{s:g}") for s in strengths}, index=idx)
        print(f"\ndetection of the test attacker under {mname}\n")
        print(t.reset_index().to_string(index=False))
    print(f"\nraw rows saved to {out}")


if __name__ == "__main__":
    main()
