"""Assemble one shot from its parts and write it out."""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

from . import dsp, environment, knobs, mechanism, receiver, sources

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
    for s in STEMS:
        stems[s] *= 10 ** (gains[s] / 20)
    # Each stem's distance back to 1 m, for feeding the space; the blast's
    # also undoes its directivity toward the listener, since the space hears
    # the gun from every side.
    d = k["blast.directivity_db"] * math.log(10) / 10
    mean_dir_db = 10 * math.log10(math.sinh(d) / d) if d > 0 else 0.0
    d_case = math.dist(bounces[0].pos, listener) if bounces else d_mech
    feed = {"blast": d_muzzle * 10 ** ((mean_dir_db - k["blast.directivity_db"] * cos_theta) / 20),
            "nwave": d_muzzle, "mech": d_mech, "case": d_case}

    log = {
        "t0": t0,
        "speed_of_sound": c,
        "mach": cy.bullet_velocity / c,
        "muzzle_exit": cy.t_exit,
        "listener": listener,
        "muzzle": muzzle,
        "feed": feed,
        "blast_peak_pa": sources.blast_level(k, cos_theta, d_muzzle)[0],
        "nwave": nw is not None,
        "events": [{"t": e.t, "name": e.name, "v": e.v} for e in cy.events],
        "casing": [{"t": b.t, "v": b.v, "end": b.end, "pos": b.pos} for b in bounces],
        "notes": cy.notes,
        "knobs": k,
    }
    return stems, log


def place(stems: dict, k, log, fs, rng) -> dict:
    """Put each stem in the space: the direct sound, its ground image, and
    the scattered field fed by the stem at 1 m. Pascals at the listener."""
    c = log["speed_of_sound"]
    h = environment.impulse_response(k, fs, rng, c)
    out = {}
    for name, x in stems.items():
        if not np.any(x):
            out[name] = np.zeros(len(x) + len(h) - 1)
            continue
        y = environment.convolve(x * log["feed"][name], h)
        if name != "case":  # the casing lies on the ground already
            y[:len(x)] += environment.ground(x, fs, k, c, log["muzzle"], log["listener"])
        y[:len(x)] += x
        out[name] = y
    return out


def trim(x: np.ndarray, fs, floor_db=-90.0) -> np.ndarray:
    """Cut the silent end, with a short fade."""
    a = np.abs(x)
    loud = np.flatnonzero(a > a.max() * 10 ** (floor_db / 20))
    end = min(len(x), (loud[-1] if len(loud) else 0) + int(0.05 * fs))
    y = x[:end].copy()
    fade = min(end, int(0.02 * fs))
    y[end - fade:] *= np.linspace(1, 0, fade)
    return y


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
    ap.add_argument("--env", choices=list(environment.ENVIRONMENTS), default="range",
                    help="the space around the shot (default: the outdoor range of the reference recording)")
    ap.add_argument("--receiver", choices=list(receiver.RECEIVERS) + ["none"], default="phone",
                    help="what hears it: a phone (fitted), an ear for a game mix, or none (pressure, 1.0 = mix.full_scale_pa)")
    ap.add_argument("--drive", type=float, default=0.0, metavar="DB",
                    help="saturate this many dB into tanh after the receiver")
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--stems", action="store_true", help="also write each layer")
    ap.add_argument("--normalize", action="store_true", help="peak-normalise each file to -1 dBFS")
    ap.add_argument("--pcm16", action="store_true", help="16-bit PCM instead of 32-bit float")
    ap.add_argument("--list", action="store_true", help="list the knobs and exit")
    a = ap.parse_args(argv)

    if a.list:
        print(knobs.describe())
        return
    values, spreads = dict(environment.ENVIRONMENTS[a.env]), {}
    if a.receiver != "none":
        values.update(receiver.RECEIVERS[a.receiver])
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

    def hear(mix, stems, k):
        """Receiver, drive and normalisation; stems ride the mix's gain."""
        if a.receiver == "none":
            scale, gain = 1 / k["mix.full_scale_pa"], None
            mix = mix * scale
            stems = {name: x * scale for name, x in stems.items()}
        else:
            # The threshold's RMS level is written at -14 dBFS.
            scale = 10 ** (-14 / 20) / (receiver.P0 * 10 ** (k["rec.agc_threshold_db"] / 20))
            mix, gain = receiver.process(mix, fs, k)
            mix = mix * scale
            stems = {name: receiver.process(x, fs, k, gain)[0] * scale for name, x in stems.items()}
        n = len(trim(mix, fs))
        result = [mix] + list(stems.values())
        for i, x in enumerate(result):
            x = x[:n].copy()
            fade = min(n, int(0.02 * fs))
            x[n - fade:] *= np.linspace(1, 0, fade)
            if a.drive > 0:
                g = 10 ** (a.drive / 20)
                x = np.tanh(g * x) / g
            result[i] = x
        if a.normalize:
            peak = np.max(np.abs(result[0])) + 1e-20
            result = [x / peak * 10 ** (-1 / 20) for x in result]
        return result[0], dict(zip(stems, result[1:]))

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
            placed = place(stems, k, log, fs, rng)
            parts.append((i * hop, sum(placed.values()), log["t0"]))
            logs.append(log)
            if vi["magazine.rounds"] == 0:
                break
        lead = -min(t0 for _, _, t0 in parts)
        n = max(p + len(x) for p, x, _ in parts) + int(lead * fs) + 1
        mix = np.zeros(n)
        for p, x, t0 in parts:
            dsp.add(mix, p + int((lead + t0) * fs), x)
        mix, _ = hear(mix, {}, knobs.draw(np.random.default_rng(0), values, {}))
        write_wav(out / "sequence.wav", mix, fs, a.pcm16)
        (out / "sequence.json").write_text(json.dumps(logs, indent=1))
        report(logs)
        return

    for i in range(a.shots):
        rng = np.random.default_rng([a.seed, i])
        k = knobs.draw(rng, values, spreads)
        stems, log = shot(k, gun, rng, fs)
        placed = place(stems, k, log, fs, rng)
        mix, placed = hear(sum(placed.values()), placed if a.stems else {}, k)
        write_wav(out / f"shot_{i:03}.wav", mix, fs, a.pcm16)
        for name, x in placed.items():
            write_wav(out / f"shot_{i:03}_{name}.wav", x, fs, a.pcm16)
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
