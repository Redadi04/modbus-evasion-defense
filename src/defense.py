#!/usr/bin/env python3
"""Compare defenses against jitter/padding evasion on flow features.

Regimes (each trained on the same clean training flows):
  baseline     train on clean flows only
  adv-train    adversarial training: add perturbed copies of the TRAINING attacker flows
               (label = malicious); perturbed copies of test flows are never used for training
  robust-K     robust features: drop the K features that the perturbations move the most
               (measured on training flows only), then train on clean flows
  adv+robust   both of the above

For every regime and model it reports, averaged over --seeds random group splits:
  F1 and false-positive rate (FPR) on clean test traffic, detection rate (DR) on the
  perturbed test attacker flows, model size and prediction time per flow.

Usage (flows exported from the clean pcap and from each perturbed pcap):
  python3 defense.py --flows flows_clean.csv --attacker 127.0.0.99 \
      --perturbed jitter=fj.csv padding=fp.csv both=fb.csv --seeds 10 --robust-drop 10

--adv-names picks which perturbations are used for adversarial training (default: all).
Train on some and test on the others to see whether a defense generalises to unseen evasion.
"""
import argparse
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from evaluate import attacker_mask, features, flow_key, keep_modbus, norm


def make_models(seed):
    return {
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "HistGradBoost": HistGradientBoostingClassifier(learning_rate=0.1, max_iter=100, random_state=seed),
    }


def load(a):
    allf = keep_modbus(norm(pd.read_csv(a.flows)), "flows")
    is_att = attacker_mask(allf, a.attacker)
    ben, mal = allf[~is_att].copy(), allf[is_att].copy()
    pert = {}
    for item in a.perturbed:
        name, path = item.split("=", 1)
        p = keep_modbus(norm(pd.read_csv(path)), f"perturbed[{name}]")
        pert[name] = p[attacker_mask(p, a.attacker)].copy()
    return ben, mal, pert


def robust_keep(X, y, keys, tr, pert, cols, k):
    """Return the columns left after dropping the k features most moved by the perturbations."""
    if not k:
        return cols, []
    base = X.iloc[tr][y[tr] == 1].copy()
    base.index = keys[tr][y[tr] == 1]
    base = base[~base.index.duplicated()]
    sd = X.iloc[tr].std().replace(0, np.nan)
    shift = pd.Series(0.0, index=cols)
    for p in pert.values():
        xp = features(p, cols)
        xp.index = flow_key(p).values
        xp = xp[~xp.index.duplicated()]
        common = base.index.intersection(xp.index)
        if len(common) == 0:
            continue
        s = ((xp.loc[common] - base.loc[common]).abs().mean() / sd).fillna(0.0)
        shift = np.maximum(shift, s)
    dropped = list(shift.sort_values(ascending=False).head(k).index)
    return [c for c in cols if c not in dropped], dropped


def run_seed(ben, mal, pert, a, seed, adv_names):
    ben_s = ben.sample(n=min(len(ben), a.ratio * len(mal)), random_state=seed)
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

    kept, dropped = robust_keep(X, y, keys, tr, pert, cols, a.robust_drop)
    regimes = {"baseline": (cols, False), "adv-train": (cols, True)}
    if a.robust_drop:
        regimes[f"robust-{a.robust_drop}"] = (kept, False)
        regimes[f"adv+robust-{a.robust_drop}"] = (kept, True)

    rows = []
    for rname, (rc, adv) in regimes.items():
        Xtr, ytr = X.iloc[tr][rc], y[tr]
        if adv:
            extra = [xp_tr[n][rc] for n in adv_names if len(xp_tr[n])]
            Xtr = pd.concat([Xtr] + extra, ignore_index=True)
            ytr = np.concatenate([ytr] + [np.ones(len(e), dtype=int) for e in extra])
        for mname, m in make_models(seed).items():
            m.fit(Xtr, ytr)
            t0 = time.perf_counter()
            pred = m.predict(X.iloc[te][rc])
            us = (time.perf_counter() - t0) / max(len(te), 1) * 1e6
            yt = y[te]
            row = {"regime": rname, "model": mname,
                   "F1": f1_score(yt, pred, zero_division=0),
                   "FPR": float(pred[yt == 0].mean()) if (yt == 0).any() else np.nan,
                   "size_KB": len(pickle.dumps(m)) / 1024, "us_per_flow": us}
            for n in pert:
                px = xp_te[n][rc]
                row[f"DR[{n}]"] = float(m.predict(px).mean()) if len(px) else np.nan
            rows.append(row)
    return rows, dropped, (len(te), int(y[te].sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flows", required=True)
    ap.add_argument("--attacker", required=True)
    ap.add_argument("--perturbed", nargs="+", required=True, help="name=path.csv pairs")
    ap.add_argument("--seeds", type=int, default=10, help="number of random group splits (default 10)")
    ap.add_argument("--robust-drop", type=int, default=0, help="K features to drop for the robust-feature regimes")
    ap.add_argument("--adv-names", default="", help="comma list of perturbation names for adversarial training (default all)")
    ap.add_argument("--ratio", type=int, default=20)
    a = ap.parse_args()

    ben, mal, pert = load(a)
    adv_names = [n for n in a.adv_names.split(",") if n] or list(pert)
    bad = [n for n in adv_names if n not in pert]
    if bad:
        raise SystemExit(f"--adv-names has unknown names: {bad}; available: {list(pert)}")
    print(f"benign={len(ben)} malicious={len(mal)}; adversarial training uses: {adv_names}")

    all_rows, first = [], True
    for s in range(a.seeds):
        rows, dropped, (n_te, n_mal_te) = run_seed(ben, mal, pert, a, s, adv_names)
        all_rows += rows
        if first:
            print(f"seed 0 test split: {n_te} flows ({n_mal_te} malicious)")
            if dropped:
                print("features dropped for the robust regimes (seed 0):", ", ".join(dropped))
            first = False

    df = pd.DataFrame(all_rows)
    metrics = [c for c in df.columns if c not in ("regime", "model")]
    g = df.groupby(["regime", "model"], sort=False)[metrics]
    mean, std = g.mean(), g.std().fillna(0.0)
    out = pd.DataFrame(index=mean.index)
    for c in metrics:
        if c in ("size_KB", "us_per_flow"):
            out[c] = mean[c].round(1)
        else:
            out[c] = [f"{m:.2f}±{s:.2f}" for m, s in zip(mean[c], std[c])]
    print(f"\nmean ± std over {a.seeds} splits\n")
    print(out.reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
