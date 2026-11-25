#!/usr/bin/env python3
"""Perturbation-strength sweep: where do jitter / padding evade the detectors, and where do
adversarial-training defenses stop working?

For each input capture the script
  1. exports flows from the clean pcap,
  2. makes perturbed copies (jitter, padding, both) at every strength in --strengths
     (strength 1 = maximum delay / padding, smaller values apply that share of it),
  3. exports flows from each perturbed pcap (cached in --workdir, so re-runs are quick),
  4. trains Decision Tree / Random Forest / Histogram Gradient Boosting under several regimes
       baseline        clean training flows only
       adv@S           clean + perturbed copies of the training attacker flows made at strength S
                       (all three perturbation kinds), for each S in --adv-strengths
     and measures the detection rate (DR) on perturbed test attacker flows at every strength.
Results are averaged over all captures x --seeds random group splits. Perturbed copies of a
test flow are never used for training.

Usage (run with the venv active; needs cicflowmeter on PATH):
  python3 sweep.py --pcaps mix1.pcap mix2.pcap mix3.pcap --attacker 127.0.0.99 \
      --workdir sweepwork --strengths 0.25,0.5,0.75,1 --adv-strengths 0.25,1 --seeds 5
"""
import argparse
import contextlib
import io
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from defense import make_models
from evaluate import attacker_mask, features, flow_key, keep_modbus, norm
from perturb_pcap import perturb

MODES = {"jitter": "static-jitter", "padding": "static-padding", "both": "static-both"}
WIN = {"window": 0.0, "block": 30.0}     # set by --window / --block; 0 = plain cicflowmeter export


def export_flows(pcap, csv):
    if os.path.exists(csv) and os.path.getsize(csv) > 0:
        return
    if WIN["window"] > 0:
        from winflows import windowed_export
        windowed_export(pcap, csv, WIN["window"], WIN["block"])
        return
    exe = shutil.which("cicflowmeter")
    if not exe:
        sys.exit("cicflowmeter not found on PATH; activate the venv where you installed it.")
    r = subprocess.run([exe, "-f", pcap, "-c", csv], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(csv) or os.path.getsize(csv) == 0:
        sys.exit(f"flow export failed for {pcap}:\n{(r.stderr or r.stdout)[-800:]}")


def prepare(pcap, a, strengths):
    base = os.path.splitext(os.path.basename(pcap))[0]
    clean_csv = os.path.join(a.workdir, f"{base}_clean.csv")
    export_flows(pcap, clean_csv)
    pert_csv = {}
    for mname, mode in MODES.items():
        for s in strengths:
            name = f"{mname}@{s:g}"
            csv = os.path.join(a.workdir, f"{base}_{mname}_{s:g}.csv")
            if not (os.path.exists(csv) and os.path.getsize(csv) > 0):
                tmp = os.path.join(a.workdir, f"{base}_{mname}_{s:g}.pcap")
                perturb(pcap, tmp, a.attacker, mode, 0, s)
                export_flows(tmp, csv)
                os.remove(tmp)
            pert_csv[name] = csv
    return clean_csv, pert_csv


def load(clean_csv, pert_csv, attacker):
    with contextlib.redirect_stdout(io.StringIO()):
        allf = keep_modbus(norm(pd.read_csv(clean_csv)), "flows")
        is_att = attacker_mask(allf, attacker)
        ben, mal = allf[~is_att].copy(), allf[is_att].copy()
        pert = {}
        for n, path in pert_csv.items():
            p = keep_modbus(norm(pd.read_csv(path)), n)
            pert[n] = p[attacker_mask(p, attacker)].copy()
    return ben, mal, pert


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
    for rname, names in regimes.items():
        Xtr, ytr = X.iloc[tr], y[tr]
        if names:
            extra = [xp_tr[n] for n in names if len(xp_tr[n])]
            Xtr = pd.concat([Xtr] + extra, ignore_index=True)
            ytr = np.concatenate([ytr] + [np.ones(len(e), dtype=int) for e in extra])
        for mname, m in make_models(seed).items():
            m.fit(Xtr, ytr)
            pred = m.predict(X.iloc[te])
            yt = y[te]
            row = {"capture": tag, "seed": seed, "regime": rname, "model": mname,
                   "F1": f1_score(yt, pred, zero_division=0),
                   "FPR": float(pred[yt == 0].mean()) if (yt == 0).any() else np.nan}
            for n in pert:
                row[n] = float(m.predict(xp_te[n]).mean()) if len(xp_te[n]) else np.nan
            rows.append(row)
    return rows


def parse_list(s):
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcaps", nargs="+", required=True)
    ap.add_argument("--attacker", required=True)
    ap.add_argument("--workdir", default="sweepwork")
    ap.add_argument("--strengths", default="0.25,0.5,0.75,1", help="perturbation strengths to test")
    ap.add_argument("--adv-strengths", default="0.25,1", help="strengths used for adversarial training")
    ap.add_argument("--seeds", type=int, default=5, help="random splits per capture")
    ap.add_argument("--ratio", type=int, default=20)
    ap.add_argument("--window", type=float, default=0.0,
                    help="cut captures into windows of this many seconds before exporting flows "
                         "(for long-lived connections); 0 = off")
    ap.add_argument("--block", type=float, default=30.0, help="group block length in seconds (with --window)")
    a = ap.parse_args()
    WIN["window"], WIN["block"] = a.window, a.block

    strengths, adv_strengths = parse_list(a.strengths), parse_list(a.adv_strengths)
    bad = [s for s in adv_strengths if s not in strengths]
    if bad:
        sys.exit(f"--adv-strengths {bad} must also appear in --strengths")
    os.makedirs(a.workdir, exist_ok=True)

    all_rows = []
    for i, pcap in enumerate(a.pcaps):
        print(f"[{i + 1}/{len(a.pcaps)}] {pcap}: exporting flows (cached in {a.workdir}) ...", flush=True)
        clean_csv, pert_csv = prepare(pcap, a, strengths)
        ben, mal, pert = load(clean_csv, pert_csv, a.attacker)
        print(f"    benign flows={len(ben)} attacker flows={len(mal)}", flush=True)
        regimes = {"baseline": []}
        for s in adv_strengths:
            regimes[f"adv@{s:g}"] = [n for n in pert if n.endswith(f"@{s:g}")]
        for seed in range(a.seeds):
            all_rows += run_split(ben, mal, pert, seed, a.ratio, regimes, os.path.basename(pcap))

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(a.workdir, "sweep_results.csv"), index=False)
    reps = df.groupby(["capture", "seed"]).ngroups
    print(f"\n{reps} repetitions ({len(a.pcaps)} captures x {a.seeds} splits); cells are mean±std of the detection rate")
    order = df[["regime", "model"]].drop_duplicates()
    idx = pd.MultiIndex.from_frame(order)

    def cell(col):
        g = df.groupby(["regime", "model"], sort=False)[col]
        return [f"{g.mean()[k]:.2f}±{g.std().fillna(0)[k]:.2f}" for k in idx]

    clean = pd.DataFrame({"F1": cell("F1"), "FPR": cell("FPR")}, index=idx)
    print("\nclean test traffic\n")
    print(clean.reset_index().to_string(index=False))
    for mname in MODES:
        t = pd.DataFrame({f"strength {s:g}": cell(f"{mname}@{s:g}") for s in strengths}, index=idx)
        print(f"\ndetection rate under {mname}\n")
        print(t.reset_index().to_string(index=False))
    print(f"\nraw rows saved to {os.path.join(a.workdir, 'sweep_results.csv')}")


if __name__ == "__main__":
    main()
