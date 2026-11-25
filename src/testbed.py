#!/usr/bin/env python3
"""Self-contained Modbus TCP testbed on the loopback interface.

Runs a Modbus server, two normal polling clients and one attacker on separate loopback
addresses, and records everything with tcpdump into one pcap. All hosts share the same
Linux TCP/IP stack, so the attacker cannot be recognised by TCP options or handshake size
(the artifact found in the ICS_PCAPS capture). Only Modbus behaviour differs:

  server    127.0.0.2   port 502, 256 holding registers that drift like sensor values
  benign    127.0.0.11, 127.0.0.12   open a connection, poll 4-10 times (~1 s apart) with large
            reads (40-100 registers), occasional small write, close, pause, repeat
  attacker  127.0.0.99   after a warm-up, bursts of 10-40 rapid small FC16 writes (1-2 registers,
            20-150 ms apart) over short connections (false-data-injection style)

The attacker sends FC16 write requests so that the padding rule in perturb_pcap.py applies.
--profile mimic makes the attacker act like a normal client (same large reads and rhythm) plus
about a third writes of 2-8 registers, so it is much harder to separate from normal traffic.
This is SYNTHETIC traffic for developing and debugging the pipeline. Do not compare its
numbers with published results on real datasets.

Run (needs root for port 502 and tcpdump):
  sudo python3 testbed.py --out ~/modbus/testbed/mix.pcap --duration 600
Then:
  python3 perturb_pcap.py mix.pcap mix_jitter.pcap --attacker 127.0.0.99 --mode static-jitter
  cicflowmeter -f mix.pcap -c flows_clean.csv        (and for each perturbed pcap)
  python3 evaluate.py --flows flows_clean.csv --attacker 127.0.0.99 --perturbed jitter=flows_jitter.csv
"""
import argparse
import os
import random
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

SERVER_IP = "127.0.0.2"
PORT = 502
BENIGN_IPS = ["127.0.0.11", "127.0.0.12"]
ATTACKER_IP = "127.0.0.99"
N_REGS = 256

regs = [0] * N_REGS
regs_lock = threading.Lock()


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


# ---------------------------------------------------------------- server
def process(fc, body):
    """Return the response PDU for a request PDU (body includes the function code)."""
    try:
        if fc == 0x03:
            addr, qty = struct.unpack(">HH", body[1:5])
            if qty < 1 or qty > 125 or addr + qty > N_REGS:
                return bytes([fc | 0x80, 2])
            with regs_lock:
                vals = regs[addr:addr + qty]
            return bytes([fc, qty * 2]) + struct.pack(">%dH" % qty, *vals)
        if fc == 0x10:
            addr, qty, _bc = struct.unpack(">HHB", body[1:6])
            data = body[6:6 + qty * 2]          # extra (padded) bytes are ignored
            if qty < 1 or addr + qty > N_REGS or len(data) < qty * 2:
                return bytes([fc | 0x80, 3])
            with regs_lock:
                regs[addr:addr + qty] = struct.unpack(">%dH" % qty, data)
            return bytes([fc]) + struct.pack(">HH", addr, qty)
        if fc == 0x06:
            addr, val = struct.unpack(">HH", body[1:5])
            if addr >= N_REGS:
                return bytes([fc | 0x80, 2])
            with regs_lock:
                regs[addr] = val
            return body[:5]
    except struct.error:
        return bytes([fc | 0x80, 3])
    return bytes([fc | 0x80, 1])


def handle(conn, stop):
    with conn:
        conn.settimeout(5)
        try:
            while not stop.is_set():
                hdr = recv_exact(conn, 7)
                if hdr is None:
                    return
                tid, _pid, length, uid = struct.unpack(">HHHB", hdr)
                body = recv_exact(conn, length - 1) if length > 1 else b""
                if body is None or not body:
                    return
                resp = process(body[0], body)
                conn.sendall(struct.pack(">HHHB", tid, 0, len(resp) + 1, uid) + resp)
        except OSError:
            return


def server(stop, ready):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((SERVER_IP, PORT))
    srv.listen(256)
    srv.settimeout(0.5)
    ready.set()
    while not stop.is_set():
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=handle, args=(conn, stop), daemon=True).start()
    srv.close()


def drift(stop, rng):
    """Make register values change slowly, like process measurements."""
    while not stop.wait(0.5):
        with regs_lock:
            for _ in range(10):
                i = rng.randrange(N_REGS)
                regs[i] = (regs[i] + rng.randint(-20, 20)) & 0xFFFF


# ---------------------------------------------------------------- clients
def connect(src):
    s = socket.socket()
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.bind((src, 0))
    s.settimeout(3)
    s.connect((SERVER_IP, PORT))
    return s


def request(s, tid, pdu):
    s.sendall(struct.pack(">HHHB", tid, 0, len(pdu) + 1, 1) + pdu)
    hdr = recv_exact(s, 7)
    if hdr is None:
        raise OSError("server closed")
    length = struct.unpack(">H", hdr[4:6])[0]
    if length > 1 and recv_exact(s, length - 1) is None:
        raise OSError("server closed")


def benign_client(ip, stop, rng, speed):
    tid = 0
    while not stop.is_set():
        try:
            s = connect(ip)
            for _ in range(rng.randint(4, 10)):
                tid = (tid + 1) & 0xFFFF
                if rng.random() < 0.05:                      # occasional setpoint write
                    qty = rng.choice([1, 2])
                    vals = struct.pack(">%dH" % qty, *[rng.randint(0, 5000) for _ in range(qty)])
                    pdu = struct.pack(">BHHB", 0x10, rng.randint(100, 120), qty, qty * 2) + vals
                else:                                        # large block read
                    pdu = struct.pack(">BHH", 0x03, 0, rng.choice([40, 60, 80, 100]))
                request(s, tid, pdu)
                if stop.wait(rng.uniform(0.8, 1.2) / speed):
                    break
            s.close()
        except OSError:
            stop.wait(0.5)
        stop.wait(rng.uniform(0.5, 2.0) / speed)


def attacker_mimic(ip, stop, rng, speed, warmup):
    """Harder attacker: behaves like a normal polling client (same connection pattern, same large
    reads, same ~1 s rhythm) but a third of its requests are FC16 writes of 2-8 registers."""
    if stop.wait(warmup):
        return
    tid = 0
    while not stop.is_set():
        try:
            s = connect(ip)
            for _ in range(rng.randint(4, 10)):
                tid = (tid + 1) & 0xFFFF
                if rng.random() < 0.33:
                    qty = rng.randint(2, 8)
                    vals = struct.pack(">%dH" % qty, *[rng.randint(0, 65535) for _ in range(qty)])
                    pdu = struct.pack(">BHHB", 0x10, rng.randint(0, 100), qty, qty * 2) + vals
                else:
                    pdu = struct.pack(">BHH", 0x03, 0, rng.choice([40, 60, 80, 100]))
                request(s, tid, pdu)
                if stop.wait(rng.uniform(0.8, 1.2) / speed):
                    break
            s.close()
        except OSError:
            stop.wait(0.5)
        stop.wait(rng.uniform(0.5, 2.0) / speed)


def attacker_stealth(ip, stop, rng, speed, warmup, frac):
    """Adaptive attacker: identical to a normal client (same reads, rhythm, connection pattern) and its
    writes copy the normal client's setpoint writes (1-2 registers at addresses 100-120). Only the
    share of writes (frac) differs from normal traffic (normal clients write about 5% of requests)."""
    if stop.wait(warmup):
        return
    tid = 0
    while not stop.is_set():
        try:
            s = connect(ip)
            for _ in range(rng.randint(4, 10)):
                tid = (tid + 1) & 0xFFFF
                if rng.random() < frac:
                    qty = rng.choice([1, 2])
                    vals = struct.pack(">%dH" % qty, *[rng.randint(0, 5000) for _ in range(qty)])
                    pdu = struct.pack(">BHHB", 0x10, rng.randint(100, 120), qty, qty * 2) + vals
                else:
                    pdu = struct.pack(">BHH", 0x03, 0, rng.choice([40, 60, 80, 100]))
                request(s, tid, pdu)
                if stop.wait(rng.uniform(0.8, 1.2) / speed):
                    break
            s.close()
        except OSError:
            stop.wait(0.5)
        stop.wait(rng.uniform(0.5, 2.0) / speed)


def attacker(ip, stop, rng, speed, warmup):
    if stop.wait(warmup):
        return
    tid = 0
    while not stop.is_set():
        try:
            s = connect(ip)
            for _ in range(rng.randint(10, 40)):
                tid = (tid + 1) & 0xFFFF
                qty = rng.choice([1, 2])
                vals = struct.pack(">%dH" % qty, *[rng.randint(0, 65535) for _ in range(qty)])
                pdu = struct.pack(">BHHB", 0x10, rng.randint(0, 100), qty, qty * 2) + vals
                request(s, tid, pdu)
                if stop.wait(rng.uniform(0.02, 0.15) / speed):
                    break
            s.close()
        except OSError:
            stop.wait(0.5)
        stop.wait(rng.uniform(1.0, 4.0) / speed)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output pcap path")
    ap.add_argument("--duration", type=float, default=600, help="seconds of traffic (default 600)")
    ap.add_argument("--warmup", type=float, default=0.2, help="fraction of the run with no attack (default 0.2)")
    ap.add_argument("--speed", type=float, default=1.0, help="time compression factor (default 1 = real time)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--write-frac", type=float, default=0.15,
                    help="stealth profile only: share of the attacker's requests that are writes")
    ap.add_argument("--profile", choices=["burst", "mimic", "stealth"], default="burst",
                    help="attacker behaviour: burst = rapid small writes (default, easy to separate); "
                         "mimic = acts like a normal client plus some writes (harder)")
    a = ap.parse_args()

    if os.geteuid() != 0:
        sys.exit("Run as root (port 502 and tcpdump need it): sudo python3 testbed.py ...")
    if not shutil.which("tcpdump"):
        sys.exit("tcpdump not found; install it (sudo apt install tcpdump).")
    out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    stop = threading.Event()
    ready = threading.Event()
    threading.Thread(target=server, args=(stop, ready), daemon=True).start()
    if not ready.wait(5):
        sys.exit("Server did not start (is port 502 already in use?).")
    threading.Thread(target=drift, args=(stop, random.Random(a.seed)), daemon=True).start()

    cap = subprocess.Popen(["tcpdump", "-i", "lo", "-U", "-w", out, "tcp", "port", str(PORT)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)                                   # let tcpdump start before traffic

    workers = []
    for k, ip in enumerate(BENIGN_IPS):
        workers.append(threading.Thread(target=benign_client,
                                        args=(ip, stop, random.Random(a.seed + 10 + k), a.speed), daemon=True))
    if a.profile == "stealth":
        workers.append(threading.Thread(target=attacker_stealth,
                                        args=(ATTACKER_IP, stop, random.Random(a.seed + 99), a.speed,
                                              a.duration * a.warmup, a.write_frac), daemon=True))
    else:
        workers.append(threading.Thread(target=attacker_mimic if a.profile == "mimic" else attacker,
                                        args=(ATTACKER_IP, stop, random.Random(a.seed + 99), a.speed,
                                              a.duration * a.warmup), daemon=True))
    for w in workers:
        w.start()
    print(f"running {a.duration:.0f}s (speed x{a.speed:g}); attack starts after {a.duration * a.warmup:.0f}s ...")
    try:
        stop.wait(a.duration)
    except KeyboardInterrupt:
        print("interrupted; stopping early")
    stop.set()
    for w in workers:
        w.join(timeout=5)
    time.sleep(1.0)
    cap.send_signal(signal.SIGINT)
    cap.wait(timeout=10)
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid and gid and os.path.exists(out):
        os.chown(out, int(uid), int(gid))
    print(f"wrote {out}")
    print(f"attacker IP: {ATTACKER_IP}   benign clients: {', '.join(BENIGN_IPS)}   server: {SERVER_IP}:{PORT}")


if __name__ == "__main__":
    main()
