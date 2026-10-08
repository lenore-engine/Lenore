"""Assemble one shot from its parts and write it out."""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

from . import dsp, knobs, mechanism, sources

STEMS = ("blast", "nwave", "mech", "case")


def geometry(k):
    gh = k["gun.height"]
    muzzle = (0.0, 0.0, gh)
    r = k["listener.distance"]
    th = math.radians(k["listener.azimuth"])
    dz = k["listener.height"] - gh
    rh = math.sqrt(max(r * r - dz * dz, 0.01))
    # The listener stands to the right of the line of fire, as the shooter's
    # right ear does; the casing goes the same way.
    listener = (rh * math.cos(th), -rh * math.sin(th), k["listener.height"])
    return muzzle, listener


def shot(k: dict, gun: sources.Gun, rng, fs: float) -> tuple[dict, dict]:
    cy = mechanism.run(k)
    c = cy.c
    muzzle, listener = geometry(k)
    mech_pos = (-0.08, 0.0, muzzle[2])
    port = (-0.06, -0.015, muzzle[2] + 0.02)
    d_muzzle = math.dist(muzzle, listener)
    d_mech = max(math.dist(mech_pos, listener), 0.1)
    rel = np.subtract(listener, muzzle)
    cos_theta = rel[0] / (np.linalg.norm(rel) + 1e-12)

    bounces, roll_start = [], 0.0
    if cy.eject_t is not None:
        bounces, roll_start = mechanism.casing_flight(k, rng, cy.eject_t, port)

    t0 = min(e.t for e in cy.events) - 0.005
    t_end = max(e.t for e in cy.events) + 0.5
    if bounces:
        t_end = max(t_end, bounces[-1].t + k["case.roll_time"] + 0.7)
    n = int((t_end - t0) * fs)
    stems = {s: np.zeros(n) for s in STEMS}

    for start, sig in sources.muzzle(k, rng, fs, cy.t_exit + d_muzzle / c, cos_theta, d_muzzle, t0):
        dsp.add(stems["blast"], start, sig)

    nw = sources.n_wave(k, rng, fs, c, cy.t_exit, np.subtract(listener, muzzle), t0)
    if nw is not None:
        dsp.add(stems["nwave"], *nw)

    for e in cy.events:
        start = int((e.t + d_mech / c - t0) * fs)
        if e.name == "da_pull":
            sig = sources.da_pull(k, gun, rng, fs)
        else:
            sig = sources.mech_event(e.name, e.v, gun, k, rng, fs)
        dsp.add(stems["mech"], start, sig / d_mech)
        if e.name == "eject":
            dsp.add(stems["case"], start, sources.eject_ring(k, rng, fs) / d_mech)
    s, rail = sources.rail_noise(cy, k, rng, fs, t0 - d_mech / c)
    dsp.add(stems["mech"], s, rail / d_mech)

    for start, sig in sources.casing(k, rng, fs, bounces, roll_start, listener, c, t0):
        dsp.add(stems["case"], start, sig)

    gains = {"blast": k["mix.blast_db"], "nwave": k["mix.nwave_db"],
             "mech": k["mix.mech_db"], "case": k["mix.case_db"]}
    fsc = k["mix.full_scale_pa"]
    for s in STEMS:
        stems[s] *= 10 ** (gains[s] / 20) / fsc

    log = {
        "t0": t0,
        "speed_of_sound": c,
        "mach": cy.bullet_velocity / c,
        "muzzle_exit": cy.t_exit,
        "listener": listener,
        "blast_peak_pa": sources.blast_level(k, cos_theta, d_muzzle)[0],
        "nwave": nw is not None,
        "events": [{"t": e.t, "name": e.name, "v": e.v} for e in cy.events],
        "casing": [{"t": b.t, "v": b.v, "end": b.end, "pos": b.pos} for b in bounces],
        "notes": cy.notes,
        "knobs": k,
    }
    return stems, log


def room(x: np.ndarray, kind: str, k, fs, rng) -> np.ndarray:
    """Audition only: a stand-in for the propagation engine, so a dry source
    can be judged in a plausible space. Not part of the source model."""
    if kind == "dry":
        return x
    if kind == "ground":
        muzzle, listener = geometry(k)
        image = (muzzle[0], muzzle[1], -muzzle[2])
        d1, d2 = math.dist(muzzle, listener), math.dist(image, listener)
        lag = int((d2 - d1) / 343.0 * fs)
        y = x.copy()
        y[lag:] += 0.8 * (d1 / d2) * x[:len(x) - lag]
        return y
    if kind == "room":
        rt60, n = 0.7, int(1.2 * fs)
        t = np.arange(n) / fs
        lo = dsp.shaped_noise(rng, n, fs, 400, 1.5) * np.exp(-6.9 * t / rt60)
        hi = dsp.shaped_noise(rng, n, fs, 4000, 1.5) * np.exp(-6.9 * t / (0.5 * rt60))
        ir = (lo + 0.5 * hi) * 0.02
        ir[0] = 1.0
        ir[int(0.004 * fs)] += 0.5
        ir[int(0.009 * fs)] += 0.35
        size = len(x) + n
        return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)
    raise ValueError(kind)


def write_wav(path: Path, x: np.ndarray, fs: int, pcm16: bool) -> None:
    if pcm16:
        data = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        fmt, bits = 1, 16
    else:
        data = x.astype("<f4").tobytes()
        fmt, bits = 3, 32
    ch, ba = 1, bits // 8
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, fmt, ch, fs, fs * ba, ba, bits))
        f.write(b"data" + struct.pack("<I", len(data)) + data)


def parse_sets(items):
    values, spreads = {}, {}
    for item in items or []:
        key, _, val = item.partition("=")
        if key not in knobs.KNOBS:
            sys.exit(f"unknown knob {key!r}; see --list")
        base, _, spread = val.partition("±")
        if not spread:
            base, _, spread = val.partition("+-")
        values[key] = float(base)
        if spread:
            spreads[key] = float(spread)
    return values, spreads


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gunshot", description="CZ 75 B shot generator")
    ap.add_argument("-o", "--out", default="out", help="output directory")
    ap.add_argument("-n", "--shots", type=int, default=1, help="number of shots")
    ap.add_argument("--seed", type=int, default=1, help="shot-to-shot variation")
    ap.add_argument("--gun-seed", type=int, default=75, help="which individual pistol")
    ap.add_argument("--preset", action="append", help="JSON file of knob values (applied in order)")
    ap.add_argument("--set", action="append", metavar="KNOB=V[±S]", help="set a knob, optionally its spread")
    ap.add_argument("--fixed", action="store_true", help="zero every spread: the knobs exactly as set")
    ap.add_argument("--sequence", type=float, metavar="SECONDS",
                    help="fire the shots into one file this far apart, emptying the magazine")
    ap.add_argument("--room", choices=["dry", "ground", "room"], default="dry",
                    help="audition space (not part of the source model)")
    ap.add_argument("--drive", type=float, default=0.0, metavar="DB",
                    help="audition: saturate this many dB into tanh, as an overloaded ear or recorder does at 155 dB")
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--stems", action="store_true", help="also write each layer")
    ap.add_argument("--normalize", action="store_true", help="peak-normalise each file to -1 dBFS")
    ap.add_argument("--pcm16", action="store_true", help="16-bit PCM instead of 32-bit float")
    ap.add_argument("--list", action="store_true", help="list the knobs and exit")
    a = ap.parse_args(argv)

    if a.list:
        print(knobs.describe())
        return
    values, spreads = {}, {}
    for p in a.preset or []:
        data = json.loads(Path(p).read_text())
        for key, val in data.items():
            if key.startswith("_"):
                continue
            if key not in knobs.KNOBS:
                sys.exit(f"{p}: unknown knob {key!r}")
            if isinstance(val, list):
                values[key], spreads[key] = float(val[0]), float(val[1])
            else:
                values[key] = float(val)
    v, s = parse_sets(a.set)
    values.update(v)
    spreads.update(s)
    if a.fixed:
        spreads = {key: 0.0 for key in knobs.KNOBS}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gun = sources.Gun(np.random.default_rng(a.gun_seed))
    fs = a.rate

    def finish(x):
        if a.drive > 0:
            g = 10 ** (a.drive / 20)
            x = np.tanh(g * x) / g
        if a.normalize:
            x = x / (np.max(np.abs(x)) + 1e-20) * 10 ** (-1 / 20)
        return x

    if a.sequence:
        rounds = values.get("magazine.rounds", knobs.KNOBS["magazine.rounds"].default)
        hop = int(a.sequence * fs)
        parts, logs = [], []
        for i in range(a.shots):
            rng = np.random.default_rng([a.seed, i])
            vi = dict(values)
            vi["magazine.rounds"] = max(rounds - i, 0)
            if i > 0:
                vi["trigger.double_action"] = 0.0
            k = knobs.draw(rng, vi, spreads)
            stems, log = shot(k, gun, rng, fs)
            parts.append((i * hop, sum(stems.values()), log["t0"]))
            logs.append(log)
            if vi["magazine.rounds"] == 0:
                break
        lead = -min(t0 for _, _, t0 in parts)
        n = max(p + len(x) for p, x, _ in parts) + int(lead * fs) + 1
        mix = np.zeros(n)
        for p, x, t0 in parts:
            dsp.add(mix, p + int((lead + t0) * fs), x)
        mix = room(mix, a.room, knobs.draw(np.random.default_rng(0), values, {}), fs,
                   np.random.default_rng(a.seed))
        write_wav(out / "sequence.wav", finish(mix), fs, a.pcm16)
        (out / "sequence.json").write_text(json.dumps(logs, indent=1))
        report(logs)
        return

    for i in range(a.shots):
        rng = np.random.default_rng([a.seed, i])
        k = knobs.draw(rng, values, spreads)
        stems, log = shot(k, gun, rng, fs)
        mix = room(sum(stems.values()), a.room, k, fs, rng)
        write_wav(out / f"shot_{i:03}.wav", finish(mix), fs, a.pcm16)
        if a.stems:
            for name, x in stems.items():
                write_wav(out / f"shot_{i:03}_{name}.wav", finish(room(x, a.room, k, fs, rng)), fs, a.pcm16)
        (out / f"shot_{i:03}.json").write_text(json.dumps(log, indent=1))
        report([log])


def report(logs):
    for log in logs:
        names = " ".join(f"{e['name']}@{(e['t'] - log['muzzle_exit']) * 1e3:+.1f}" for e in log["events"])
        crack = "crack" if log["nwave"] else "no crack"
        print(f"M={log['mach']:.3f} {crack}, blast {20 * math.log10(log['blast_peak_pa'] / 2e-5):.0f} dB SPL, "
              f"{len(log['casing'])} casing bounces | {names} ms")
        for note in log["notes"]:
            print("  note:", note)
