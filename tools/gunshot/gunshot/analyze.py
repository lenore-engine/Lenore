"""Measure shots in a recording, and optionally a render beside it.

    python3 -m gunshot.analyze real.wav [--synth out/shot_000.wav] [--plot cmp.png]

For each shot: peak, clipping, rise time, positive-phase duration, the
envelope peaks after the blast (the mechanism's events), high-band ticks
later on (the brass), the decay of the space, and the 1/3-octave spectrum of
the first 5 ms and of the whole shot.
"""

import argparse
import struct
import sys

import numpy as np


def read_wav(path):
    data = open(path, "rb").read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        sys.exit(f"{path}: not a RIFF/WAVE file (convert with ffmpeg -i in.x out.wav)")
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, fs, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: the real tag is in the GUID
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, ch, fs, bits)
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)
    tag, ch, fs, bits = fmt
    if tag == 3:
        x = np.frombuffer(raw, "<f4" if bits == 32 else "<f8").astype(float)
    elif bits == 16:
        x = np.frombuffer(raw, "<i2") / 32768.0
    elif bits == 24:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 1 << 23, v - (1 << 24), v) / float(1 << 23)
    elif bits == 32:
        x = np.frombuffer(raw, "<i4") / 2147483648.0
    else:
        sys.exit(f"{path}: unsupported {bits}-bit format {tag}")
    x = x[: len(x) // ch * ch].reshape(-1, ch)
    # The loudest channel; channels of a stereo pair may differ in polarity.
    return x[:, np.argmax(np.abs(x).max(axis=0))], fs


def onsets(x, fs, min_gap=0.08, rel=0.3):
    env = np.abs(x)
    thr = rel * env.max()
    out, last = [], -1e9
    for i in np.flatnonzero(env > thr):
        if i - last > min_gap * fs:
            out.append(i)
        last = i
    # Step back to where the front starts rising.
    starts = []
    for i in out:
        floor = 0.05 * env[i:i + int(0.002 * fs)].max()
        j = i
        while j > 0 and env[j - 1] > floor and i - j < int(0.001 * fs):
            j -= 1
        starts.append(j)
    return starts


def third_octaves(x, fs):
    centres = 1000 * 2.0 ** (np.arange(-17, 14) / 3)  # 20 Hz .. 20 kHz
    centres = centres[centres < 0.45 * fs]
    n = max(len(x), int(0.05 * fs))  # pad short windows for low bands
    spec = np.abs(np.fft.rfft(x, n)) ** 2 / fs
    f = np.fft.rfftfreq(n, 1 / fs)
    bands = []
    for fc in centres:
        sel = (f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))
        bands.append(10 * np.log10(spec[sel].sum() * 2 / fs + 1e-30))
    return centres, np.array(bands)


def envelope(x, fs, ms):
    n = max(1, int(ms * 1e-3 * fs))
    return np.sqrt(np.convolve(x * x, np.ones(n) / n, mode="same"))


def band(x, fs, lo, hi):
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, len(x))


def local_peaks(env, fs, start, stop, min_gap_ms, rel_db):
    a, b = int(start * fs), min(int(stop * fs), len(env))
    seg = env[a:b]
    if len(seg) < 3:
        return []
    ref = env.max()
    gap = int(min_gap_ms * 1e-3 * fs)
    idx = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] > seg[i + 1]]
    idx = [i for i in idx if 20 * np.log10(seg[i] / ref + 1e-30) > rel_db]
    picked = []
    for i in sorted(idx, key=lambda i: -seg[i]):
        if all(abs(i - j) > gap for j in picked):
            picked.append(i)
    return sorted((a + i) / fs for i in picked)


def measure(x, fs, i0, length):
    seg = x[i0:i0 + int(length * fs)]
    pk_i = int(np.argmax(np.abs(seg[: int(0.005 * fs)])))
    pk = seg[pk_i]
    sign = np.sign(pk)
    s = seg * sign  # make the front positive
    full = np.abs(x).max()
    clipped = int(np.sum(np.abs(seg[: int(0.01 * fs)]) > 0.999 * full))
    a = s[:pk_i + 1]
    r10 = np.argmax(a >= 0.1 * s[pk_i])
    r90 = np.argmax(a >= 0.9 * s[pk_i])
    zc = pk_i + int(np.argmax(s[pk_i:] < 0))
    neg = s[zc:zc + int(0.01 * fs)]
    early = third_octaves(seg[: int(0.005 * fs)], fs)
    whole = third_octaves(seg, fs)

    env = envelope(seg, fs, 0.5)
    mech = local_peaks(env, fs, 0.004, 0.06, 2.0, -45)
    hi = envelope(band(seg, fs, 6000, 14000), fs, 1.0)
    brass = local_peaks(hi, fs, 0.25, length, 40.0, -60)

    # Schroeder decay of the whole shot, fitted from -5 to -25 dB.
    e = np.cumsum((seg ** 2)[::-1])[::-1]
    edc = 10 * np.log10(e / e[0] + 1e-30)
    t5, t25 = np.argmax(edc < -5), np.argmax(edc < -25)
    rt60 = 3 * (t25 - t5) / fs if t25 > t5 else float("nan")

    return {
        "peak_dbfs": 20 * np.log10(abs(pk) + 1e-30),
        "polarity": "+" if sign > 0 else "-",
        "clipped_samples": clipped,
        "rise_us": (r90 - r10) / fs * 1e6,
        "positive_ms": (zc - r10) / fs * 1e3 if zc > pk_i else float("nan"),
        "negative_peak_rel": float(-neg.min() / s[pk_i]) if len(neg) else float("nan"),
        "rt60_s": rt60,
        "mech_ms": [round(t * 1e3, 1) for t in mech],
        "brass_s": [round(t, 3) for t in brass],
        "spec_early": early,
        "spec_whole": whole,
        "env": envelope(seg, fs, 0.25),
    }


def report(name, fs, shots):
    print(f"== {name}: {fs} Hz, {len(shots)} shot(s)")
    for k, m in enumerate(shots):
        clip = f", CLIPPED {m['clipped_samples']} samples" if m["clipped_samples"] > 2 else ""
        print(f" shot {k}: peak {m['peak_dbfs']:.1f} dBFS ({m['polarity']}){clip}; rise {m['rise_us']:.0f} µs "
              f"(resolution {1e6 / fs:.0f}); positive phase {m['positive_ms']:.2f} ms; "
              f"negative/positive peak {m['negative_peak_rel']:.2f}; RT60 ≈ {m['rt60_s']:.2f} s")
        print(f"   envelope peaks 4-60 ms (mechanism, reflections): {m['mech_ms']}")
        print(f"   6-14 kHz ticks after 0.25 s (brass): {m['brass_s']}")
        c, e = m["spec_early"]
        _, w = m["spec_whole"]
        top = np.max(w)
        row = " ".join(f"{fc:>6.0f}" for fc in c[::2])
        print("   1/3-oct (every other) Hz :", row)
        print("   first 5 ms, dB re max    :", " ".join(f"{v - top:6.1f}" for v in e[::2]))
        print("   whole shot, dB re max    :", " ".join(f"{v - top:6.1f}" for v in w[::2]))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gunshot.analyze")
    ap.add_argument("wav")
    ap.add_argument("--synth", help="a render to measure the same way")
    ap.add_argument("--length", type=float, default=1.5, help="seconds analysed per shot")
    ap.add_argument("--plot", help="write a comparison figure (needs matplotlib)")
    a = ap.parse_args(argv)

    sets = []
    for path in [a.wav] + ([a.synth] if a.synth else []):
        x, fs = read_wav(path)
        shots = [measure(x, fs, i, a.length) for i in onsets(x, fs)]
        report(path, fs, shots)
        sets.append((path, x, fs, onsets(x, fs), shots))

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(12, 11))
        for path, x, fs, ons, shots in sets:
            i0, m = ons[0], shots[0]
            seg = x[i0 - int(0.001 * fs): i0 + int(0.01 * fs)]
            seg = seg / (np.abs(seg).max() + 1e-30) * (1 if m["polarity"] == "+" else -1)
            ax[0].plot((np.arange(len(seg)) / fs - 0.001) * 1e3, seg, lw=0.8, label=path)
            env = m["env"]
            ax[1].plot(np.arange(len(env)) / fs * 1e3, 20 * np.log10(env / env.max() + 1e-9), lw=0.6, label=path)
            c, w = m["spec_whole"]
            ax[2].semilogx(c, w - w.max(), marker="o", label=path + " (whole)")
            c, e = m["spec_early"]
            ax[2].semilogx(c, e - w.max(), ls="--", label=path + " (first 5 ms)")
        ax[0].set(title="front, normalised", xlabel="ms")
        ax[1].set(title="envelope, dB", xlabel="ms", ylim=(-80, 3), xlim=(0, a.length * 1e3))
        ax[2].set(title="1/3-octave energy, dB re loudest band", xlabel="Hz", ylim=(-70, 3))
        for axis in ax:
            axis.legend(fontsize=7)
            axis.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(a.plot, dpi=80)


if __name__ == "__main__":
    main()
