"""Fit the space and the receiver to a recording of one shot.

    python3 -m gunshot.fit real.wav [--preset P] [--set KNOB=V] [--evals 1500] [-o fitted.json]

The gun stays as modelled; only the env.* and rec.* knobs move, within
plausible bounds, to minimise gunshot.analyze's distance plus the
differences in crest factor and kurtosis. The result is a preset: render
with it to hear the gun in that place, through that recorder. Set the
listener's distance and angle to the recording's first, with --set.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import analyze, environment, knobs, receiver, render, sources

# knob: (low, high, log scale)
FIT = {
    "env.near_db": (-40, 0, False),
    "env.near_rt": (0.1, 3.0, True),
    "env.near_rt_hf": (0.0, 1.5, False),
    "env.near_build_ms": (0.5, 100, True),
    "env.near_lo_hz": (50, 800, True),
    "env.near_tilt": (-9, 3, False),
    "env.echo_db": (-60, -5, False),
    "env.echo_width_ms": (5, 150, True),
    "env.far_db": (-60, -5, False),
    "env.far_rt": (0.3, 6, True),
    "rec.clip_db": (115, 145, False),
    "rec.highpass_hz": (20, 400, True),
    "rec.highpass_order": (1, 4, False),
    "rec.agc_threshold_db": (60, 130, False),
    "rec.agc_ratio": (1.5, 50, True),
    "rec.agc_window_ms": (0.5, 50, True),
    "rec.agc_release": (20, 3000, True),
    "rec.agc_lookahead_ms": (0, 5, False),
}


def to_unit(path, v):
    lo, hi, log = FIT[path]
    return (math.log(v / lo) / math.log(hi / lo)) if log else (v - lo) / (hi - lo)


def from_unit(path, u):
    lo, hi, log = FIT[path]
    u = min(max(u, 0.0), 1.0)
    return lo * (hi / lo) ** u if log else lo + u * (hi - lo)


def nelder_mead(f, x0, step, evals):
    """Adaptive Nelder-Mead (Gao and Han 2012) on the unit cube."""
    n = len(x0)
    a, g, c, s = 1.0, 1 + 2 / n, 0.75 - 1 / (2 * n), 1 - 1 / n
    pts = [np.clip(x0, 0, 1)] + [np.clip(x0 + step * e, 0, 1) for e in np.eye(n)]
    vals = [f(p) for p in pts]
    used = len(pts)
    while used < evals:
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        mid = np.mean(pts[:-1], axis=0)
        r = np.clip(mid + a * (mid - pts[-1]), 0, 1)
        fr = f(r)
        used += 1
        if fr < vals[0]:
            e = np.clip(mid + g * (r - mid), 0, 1)
            fe = f(e)
            used += 1
            pts[-1], vals[-1] = (e, fe) if fe < fr else (r, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = r, fr
        else:
            inside = fr >= vals[-1]
            k = np.clip(mid + (-c if inside else c) * ((pts[-1] if inside else r) - mid), 0, 1)
            fk = f(k)
            used += 1
            if fk < min(fr, vals[-1]):
                pts[-1], vals[-1] = k, fk
            else:
                for i in range(1, n + 1):
                    pts[i] = pts[0] + s * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
                used += n
        if max(vals) - min(vals) < 1e-4:
            break
    best = int(np.argmin(vals))
    return pts[best], vals[best]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gunshot.fit")
    ap.add_argument("wav")
    ap.add_argument("--preset", action="append", help="JSON knob values to start from")
    ap.add_argument("--set", action="append", metavar="KNOB=V", help="fix a knob (geometry, air)")
    ap.add_argument("--evals", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--out", default="fitted.json")
    a = ap.parse_args(argv)

    values = {}
    for p in a.preset or []:
        for key, val in json.loads(Path(p).read_text()).items():
            if not key.startswith("_"):
                values[key] = float(val[0] if isinstance(val, list) else val)
    v, _ = render.parse_sets(a.set)
    values.update(v)

    x_ref, fs = analyze.read_wav(a.wav)
    start = analyze.onsets(x_ref, fs)
    if not start:
        sys.exit(f"{a.wav}: no shot found")
    ref = analyze.bandgram(x_ref, fs, start[0])
    ref_crest, ref_kurt = analyze.crest_kurtosis(x_ref, fs, start[0])

    rng = np.random.default_rng([a.seed, 0])
    k = knobs.draw(rng, values, {key: 0.0 for key in knobs.KNOBS})
    stems, log = render.shot(k, sources.Gun(np.random.default_rng(75)), rng, fs)
    # The space is linear: three sums of the stems are all it needs.
    direct = sum(stems.values())
    grounded = direct - stems["case"]
    fed = sum(x * log["feed"][name] for name, x in stems.items())
    c = log["speed_of_sound"]

    def hear(kk):
        h = environment.impulse_response(kk, fs, np.random.default_rng(7), c)
        y = environment.convolve(fed, h)
        y[:len(direct)] += direct + environment.ground(grounded, fs, kk, c, log["muzzle"], log["listener"])
        y, _ = receiver.process(y, fs, kk)
        return y

    def cost(u, verbose=False):
        kk = dict(k)
        kk.update({path: from_unit(path, ui) for path, ui in zip(FIT, u)})
        y = hear(kk)
        on = analyze.onsets(y, fs)
        if not on:
            return 1e3
        total, per = analyze.score(analyze.distance(ref, analyze.bandgram(y, fs, on[0])))
        crest, kurt = analyze.crest_kurtosis(y, fs, on[0])
        if verbose:
            print(f"   distance {total:.2f} dB by window " + " ".join(f"{p:.1f}" for p in per)
                  + f"; crest {crest:.1f} dB (recording {ref_crest:.1f}), kurtosis {kurt:.2f} ({ref_kurt:.2f})")
        return total + 0.3 * abs(crest - ref_crest) + min(abs(kurt - ref_kurt), 10.0)

    u0 = np.array([to_unit(path, min(max(k[path], FIT[path][0]), FIT[path][1])) for path in FIT])
    print("start:")
    cost(u0, True)
    u, _ = nelder_mead(cost, u0, 0.15, a.evals // 2)
    u, _ = nelder_mead(cost, u, 0.05, a.evals // 2)  # restart: NM stalls in 16 dimensions
    print("fitted:")
    cost(u, True)
    fitted = {path: round(from_unit(path, ui), 4) for path, ui in zip(FIT, u)}
    fitted = {"_": f"fitted to {Path(a.wav).name} by gunshot.fit", **{key: values[key] for key in values}, **fitted}
    Path(a.out).write_text(json.dumps(fitted, indent=1) + "\n")
    for path in FIT:
        print(f"  {path:22} {fitted[path]:g}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
