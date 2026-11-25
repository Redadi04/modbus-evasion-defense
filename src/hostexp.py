#!/usr/bin/env python3
"""Host-level features (writes per source host over a trailing window) vs flow-only features.
Trains on --train-pcaps, tests on --test-pcaps; needs <workdir>/<name>_clean.csv cached by crosstest.py.
Regimes: flow-only, flow+host, host-only. Reports recall on the attacker and FPR on normal flows."""
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scapy.all import IP, TCP, PcapReader

from defense import make_models
from evaluate import attacker_mask, features, find, ip_cols, keep_modbus, norm, port_cols

WRITE_FC = {5, 6, 15, 16}
HOST_COLS = ["host_reqs", "host_writes", "host_write_share", "host_conns"]


def read_requests(pcap):
    """Return {(src_ip, src_port): [(time, is_write), ...]} for client requests to port 502."""
    conns = defaultdict(list)
    for p in PcapReader(pcap):
        if not (p.haslayer(IP) and p.haslayer(TCP)) or p[TCP].dport != 502:
            continue
        pay = bytes(p[TCP].payload)
        if len(pay) < 8:
            continue
        conns[(p[IP].src, p[TCP].sport)].append((float(p.time), pay[7] in WRITE_FC))
    return conns


def host_features(flows, conns, window):
    """Add the host features to a flow table (flows come from the same pcap as conns)."""
    si, di = ip_cols(flows)
    sp, dp = port_cols(flows)
    by_host = defaultdict(list)                  # host -> [(time, is_write, port)]
    for (ip, port), reqs in conns.items():
        for t, w in reqs:
            by_host[ip].append((t, w, port))
    for ip in by_host:
        by_host[ip].sort()
    times = {ip: np.array([x[0] for x in v]) for ip, v in by_host.items()}

    rows = []
    for _, r in flows.iterrows():
        # the client side is the one that is not port 502
        if int(float(r[dp])) == 502:
            ip, port = str(r[si]), int(float(r[sp]))
        else:
            ip, port = str(r[di]), int(float(r[dp]))
        mine = conns.get((ip, port))
        if not mine:
            rows.append((0, 0, 0.0, 0))
            continue
        end = max(t for t, _ in mine)
        lst, ts = by_host[ip], times[ip]
        lo, hi = np.searchsorted(ts, end - window, side="right"), np.searchsorted(ts, end, side="right")
        seg = lst[lo:hi]
        n, w = len(seg), sum(1 for x in seg if x[1])
        rows.append((n, w, w / n if n else 0.0, len({x[2] for x in seg})))
    out = flows.copy()
    out[HOST_COLS] = pd.DataFrame(rows, columns=HOST_COLS, index=flows.index)
    return out


def load(pcaps, workdir, attacker, window):
    bens, mals = [], []
    for p in pcaps:
        base = os.path.splitext(os.path.basename(p))[0]
        csv = os.path.join(workdir, f"{base}_clean.csv")
        if not os.path.exists(csv):
            raise SystemExit(f"{csv} not found: run crosstest.py or sweep.py on {p} first")
        flows = keep_modbus(norm(pd.read_csv(csv)), base)
        flows = host_features(flows, read_requests(p), window)
        is_att = attacker_mask(flows, attacker)
        bens.append(flows[~is_att])
        mals.append(flows[is_att])
        print(f"  {base}: {int((~is_att).sum())} normal, {int(is_att.sum())} attacker flows", flush=True)
    return pd.concat(bens, ignore_index=True), pd.concat(mals, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-pcaps", nargs="+", required=True)
    ap.add_argument("--train-work", required=True)
    ap.add_argument("--test-pcaps", nargs="+", required=True)
    ap.add_argument("--test-work", required=True)
    ap.add_argument("--attacker", required=True)
    ap.add_argument("--host-window", type=float, default=30.0, help="trailing window in seconds (default 30)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ratio", type=int, default=20)
    a = ap.parse_args()

    print("training captures:")
    tr_ben, tr_mal = load(a.train_pcaps, a.train_work, a.attacker, a.host_window)
    print("test captures:")
    te_ben, te_mal = load(a.test_pcaps, a.test_work, a.attacker, a.host_window)
    print("\nhost features, mean per class (test captures)")
    print(pd.DataFrame({"normal": te_ben[HOST_COLS].mean(), "attacker": te_mal[HOST_COLS].mean()}).round(3).to_string())

    rows = []
    for seed in range(a.seeds):
        ben = tr_ben.sample(n=min(len(tr_ben), a.ratio * len(tr_mal)), random_state=seed)
        data = pd.concat([ben.assign(_y=0), tr_mal.assign(_y=1)], ignore_index=True)
        y = data["_y"].values
        X = features(data.drop(columns=["_y"]))
        cols = list(X.columns)
        flow_cols = [c for c in cols if c not in HOST_COLS]
        Xb, Xm = features(te_ben, cols), features(te_mal, cols)
        for rname, rc in {"flow-only": flow_cols, "flow+host": cols, "host-only": HOST_COLS}.items():
            for mname, m in make_models(seed).items():
                m.fit(X[rc], y)
                rows.append({"regime": rname, "model": mname, "features": len(rc),
                             "recall": float(m.predict(Xm[rc]).mean()), "FPR": float(m.predict(Xb[rc]).mean())})
    df = pd.DataFrame(rows)
    g = df.groupby(["regime", "model"], sort=False)
    mean, std = g[["recall", "FPR"]].mean(), g[["recall", "FPR"]].std().fillna(0)
    out = pd.DataFrame({"features": g["features"].first(),
                        "recall (attacker)": [f"{mean.loc[k, 'recall']:.2f}+-{std.loc[k, 'recall']:.2f}" for k in mean.index],
                        "FPR (normal)": [f"{mean.loc[k, 'FPR']:.2f}+-{std.loc[k, 'FPR']:.2f}" for k in mean.index]})
    print(f"\n{a.seeds} training runs; test set fixed ({len(te_mal)} attacker, {len(te_ben)} normal flows)\n")
    print(out.reset_index().to_string(index=False))
    path = os.path.join(a.test_work, "hostexp_results.csv")
    df.to_csv(path, index=False)
    print(f"\nraw rows saved to {path}")


if __name__ == "__main__":
    main()
