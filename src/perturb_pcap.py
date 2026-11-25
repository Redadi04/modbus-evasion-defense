#!/usr/bin/env python3
"""Apply jitter and/or padding to the attacker's Modbus TCP packets in a pcap.

Re-implementation of the perturbation method described in Galli et al. (2026),
"Evading ML NIDS for Modbus TCP with Problem-Space Perturbations", Sections 4.2-4.3.
Written from the paper's description, NOT from the authors' code.

Usage:
  python3 perturb_pcap.py in.pcap out.pcap --attacker 185.175.0.5 --mode static-jitter
Modes: random-jitter, static-jitter, random-padding, static-padding, static-both
"""
import argparse
import random
import struct

from scapy.all import IP, TCP, Raw, PcapReader, wrpcap

MODBUS_PORT = 502
MAX_PDU = 253          # Modbus TCP: 7-byte MBAP header + 253-byte PDU = 260-byte ADU
MIN_SEP = 1e-6         # minimum gap kept between consecutive packets (seconds)
READ_FCS = {0x01, 0x02, 0x03, 0x04}   # responses carry: byte count + data
WRITE_FCS = {0x0F, 0x10}              # requests carry: ... byte count + data
EVEN_FCS = {0x03, 0x04, 0x10}         # register functions: byte count stays even


def is_attacker_modbus(pkt, attacker):
    return (
        pkt.haslayer(IP) and pkt.haslayer(TCP)
        and pkt[IP].src == attacker
        and MODBUS_PORT in (pkt[TCP].sport, pkt[TCP].dport)
    )


def pad_packet(pkt, strength=1.0):
    """Append 0x00 bytes (strength x the room left up to the max PDU size).
    Returns True if the packet changed."""
    if not pkt.haslayer(Raw):
        return False
    data = bytes(pkt[Raw].load)
    if len(data) < 8:
        return False
    tid, pid, length, uid, fc = struct.unpack(">HHHBB", data[:8])
    from_server = pkt[TCP].sport == MODBUS_PORT
    # only pad messages that carry variable-length data
    if from_server and fc in READ_FCS:
        count_at = 8
    elif (not from_server) and fc in WRITE_FCS:
        count_at = 12   # fc(7) + addr(8-9) + qty(10-11) -> byte count at offset 12
    else:
        return False
    if len(data) <= count_at:
        return False
    pdu_len = len(data) - 7
    room = int((MAX_PDU - pdu_len) * strength)
    if fc in EVEN_FCS and room % 2:
        room -= 1
    if room <= 0:
        return False
    old_count = data[count_at]
    if old_count + room > 255:
        return False
    new = bytearray(data) + bytes(room)
    new[count_at] = old_count + room
    struct.pack_into(">H", new, 4, length + room)
    pkt[Raw].load = bytes(new)
    # let scapy recompute lengths/checksums on write
    del pkt[IP].len
    del pkt[IP].chksum
    del pkt[TCP].chksum
    return True


def perturb(in_path, out_path, attacker, mode, seed, strength=1.0):
    """strength in (0, 1]: share of the maximum delay / padding that is applied (1 = the paper's setting)."""
    rng = random.Random(seed)
    pkts = [p for p in PcapReader(in_path)]
    pkts.sort(key=lambda p: float(p.time))          # keep original order (stable)
    times = [float(p.time) for p in pkts]
    do_jitter = "jitter" in mode or mode.endswith("both")
    do_pad = "padding" in mode or mode.endswith("both")
    static = mode.startswith("static")
    n_j = n_p = n_m = 0
    for i, p in enumerate(pkts):
        if not is_attacker_modbus(p, attacker):
            continue
        n_m += 1
        if do_jitter and i + 1 < len(pkts):
            room = (times[i + 1] - times[i]) - MIN_SEP
            if room > 0:
                delta = strength * room if static else rng.uniform(0, strength * room)
                p.time = times[i] + delta
                n_j += 1
        if do_pad and (static or rng.random() < 0.5):
            n_p += pad_packet(p, strength)
    wrpcap(out_path, pkts)
    print(f"{in_path}: {len(pkts)} packets, {n_m} attacker Modbus packets, "
          f"{n_j} delayed, {n_p} padded -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--attacker", required=True, help="IP of the attacker-controlled host")
    ap.add_argument("--mode", required=True,
                    choices=["random-jitter", "static-jitter", "random-padding", "static-padding", "static-both"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strength", type=float, default=1.0,
                    help="share of the maximum delay/padding to apply, in (0, 1]; default 1")
    a = ap.parse_args()
    if not 0 < a.strength <= 1:
        ap.error("--strength must be in (0, 1]")
    perturb(a.input, a.output, a.attacker, a.mode, a.seed, a.strength)
