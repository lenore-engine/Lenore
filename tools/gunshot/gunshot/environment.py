"""Where the shot is heard: ground, the scattered field around the shooter,
a distant echo and the terrain rolling it on. Linear, numpy only.

A gunshot close to the gun is mostly not the gun. The blast is a 0.1 ms
spike; what a listener hears as its body and tail is the same spike scattered
back by the ground, trees, walls and hills over the next second. So the space
is rendered here, as impulse responses fed by the dry sources at 1 m.

Every level is an energy gain re the free-field source at 1 m, so the
direct-to-scattered balance follows the listener's distance on its own.
"""

import math

import numpy as np

P_REF = 101.325e3


def air_absorption(f: np.ndarray, celsius: float, humidity: float) -> np.ndarray:
    """ISO 9613-1 pure-tone absorption of the atmosphere, dB/m."""
    T, T0, T01 = celsius + 273.15, 293.15, 273.16
    psat = 10 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151)
    h = humidity * psat
    frO = 24 + 4.04e4 * h * (0.02 + h) / (0.391 + h)
    frN = (T / T0) ** -0.5 * (9 + 280 * h * math.exp(-4.170 * ((T / T0) ** (-1 / 3) - 1)))
    f2 = np.asarray(f, float) ** 2
    return 8.686 * f2 * (1.84e-11 * (T / T0) ** 0.5 + (T / T0) ** -2.5 * (
        0.01275 * math.exp(-2239.1 / T) / (frO + f2 / frO)
        + 0.1068 * math.exp(-3352.0 / T) / (frN + f2 / frN)))


def ground_reflection(f: np.ndarray, sigma_kpa: float, cos_inc: float) -> np.ndarray:
    """Plane-wave reflection coefficient of a locally reacting ground,
    Delany-Bazley impedance from its flow resistivity (kPa s/m^2)."""
    if sigma_kpa <= 0:
        return np.zeros_like(f, dtype=complex)
    X = 1.2 * np.maximum(f, 1.0) / (sigma_kpa * 1e3)
    z = 1 + 0.0571 * X ** -0.754 - 1j * 0.087 * X ** -0.732
    zc = z * max(cos_inc, 1e-3)
    return (zc - 1) / (zc + 1)


def band_split(n: int, fs: float, centres: np.ndarray):
    """Raised-cosine bands on log frequency that sum to one."""
    F = np.fft.rfftfreq(n, 1 / fs)
    lf = np.log2(np.maximum(F, 1.0))
    lc = np.log2(centres)
    W = np.zeros((len(centres), len(F)))
    for i, c in enumerate(lc):
        lo = lc[i - 1] if i else c - 1
        hi = lc[i + 1] if i < len(lc) - 1 else c + 1
        up, dn = (lf >= lo) & (lf <= c), (lf > c) & (lf <= hi)
        W[i, up] = np.sin(0.5 * np.pi * (lf[up] - lo) / (c - lo)) ** 2
        W[i, dn] = np.cos(0.5 * np.pi * (lf[dn] - c) / (hi - c)) ** 2
        if i == 0:
            W[i, lf < c] = 1.0
        if i == len(lc) - 1:
            W[i, lf > c] = 1.0
    return F, W


CENTRES = 1000 * 2.0 ** np.arange(-4, 5)  # 62.5 Hz .. 16 kHz octaves


def diffuse(rng, n, fs, c, level_db, rt, rt_hf, build, lo_hz, tilt, air_db_m, start=0.0):
    """A diffuse field: noise whose every octave builds up over ``build`` s,
    decays with its own RT60 and loses what the air takes over the path c*t.
    Scaled so that its energy gain at 1 kHz is ``level_db`` (times the
    spectral weight: 12 dB/oct below ``lo_hz``, ``tilt`` dB/oct above 1 kHz)."""
    F, W = band_split(n, fs, CENTRES)
    t = np.maximum(np.arange(n) / fs - start, 0.0)
    on = np.arange(n) / fs >= start
    white = np.fft.rfft(rng.standard_normal(n))
    absorb = air_absorption(CENTRES, *air_db_m)
    h = np.zeros(n)
    for i, fc in enumerate(CENTRES):
        if fc > 0.45 * fs:
            break
        band_rt = rt * (1000 / fc) ** rt_hf if fc > 1000 else rt
        env = on * (1 - np.exp(-t / max(build, 1e-4))) * np.exp(-6.9 * t / band_rt)
        env *= 10 ** (-absorb[i] * c * (t + start) / 20)
        band = np.fft.irfft(white * W[i], n)
        band /= np.sqrt(np.mean(band ** 2)) + 1e-30
        weight = 10 ** (tilt * math.log2(max(fc, 1000) / 1000) / 10) / (1 + (lo_hz / fc) ** 4)
        # Energy gain of band-limited unit noise times env, per Hz of band.
        bw = fc * (2 ** 0.5 - 2 ** -0.5)
        energy = np.sum(env ** 2) / fs
        h += band * env * math.sqrt(10 ** (level_db / 10) * weight * 2 * bw / (energy * fs * fs + 1e-30))
    return h


def echo(rng, n, fs, c, delay, level_db, width, air_db_m):
    """One diffuse return from a distant face: a short noise burst at
    ``delay``, its highs taken by the air over the path."""
    if delay <= 0 or delay * fs >= n:
        return np.zeros(n)
    m = int(8 * width * fs) | 1
    tt = (np.arange(m) - m // 2) / fs
    burst = rng.standard_normal(m) * np.exp(-0.5 * (tt / width) ** 2)
    burst *= 10 ** (level_db / 20) / math.sqrt(np.sum(burst ** 2))
    h = np.zeros(n)
    i = int(delay * fs) - m // 2
    a, b = max(i, 0), min(i + m, n)
    h[a:b] = burst[a - i:b - i]
    F = np.fft.rfftfreq(n, 1 / fs)
    return np.fft.irfft(np.fft.rfft(h) * 10 ** (-air_absorption(F, *air_db_m) * c * delay / 20), n)


def impulse_response(k, fs, rng, c, length=2.5) -> np.ndarray:
    """Everything but the direct path and the ground, re the source at 1 m."""
    n = 1 << int(math.ceil(math.log2(length * fs)))
    air = (k["air.temperature"], k["air.humidity"])
    h = np.zeros(n)
    if k["env.near_db"] > -80:
        h += diffuse(rng, n, fs, c, k["env.near_db"], k["env.near_rt"], k["env.near_rt_hf"],
                     k["env.near_build_ms"] * 1e-3, k["env.near_lo_hz"], k["env.near_tilt"], air)
    d = k["env.echo_delay"]
    if d > 0 and k["env.echo_db"] > -80:
        h += echo(rng, n, fs, c, d, k["env.echo_db"], k["env.echo_width_ms"] * 1e-3, air)
    if d > 0 and k["env.far_db"] > -80:
        h += diffuse(rng, n, fs, c, k["env.far_db"], k["env.far_rt"], 0.0,
                     k["env.echo_width_ms"] * 1e-3, k["env.near_lo_hz"], k["env.near_tilt"], air, start=d)
    return h


def ground(x: np.ndarray, fs, k, c, muzzle, listener) -> np.ndarray:
    """The ground's image of ``x`` (a signal at the listener from a source
    at ``muzzle``): delayed, attenuated by the longer path, filtered by the
    ground's impedance and the air."""
    sigma = k["env.ground_kpa"]
    if sigma <= 0:
        return np.zeros_like(x)
    image = (muzzle[0], muzzle[1], -muzzle[2])
    d1, d2 = math.dist(muzzle, listener), math.dist(image, listener)
    cos_inc = (muzzle[2] + listener[2]) / d2
    n = 1 << int(math.ceil(math.log2(len(x) + fs * 0.05)))
    F = np.fft.rfftfreq(n, 1 / fs)
    air = air_absorption(F, k["air.temperature"], k["air.humidity"]) * (d2 - d1)
    H = ground_reflection(F, sigma, cos_inc) * (d1 / d2) * 10 ** (-air / 20)
    H = H * np.exp(-2j * np.pi * F * (d2 - d1) / c)
    return np.fft.irfft(np.fft.rfft(x, n) * H, n)[:len(x)]


def convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = 1 << int(math.ceil(math.log2(len(x) + len(h))))
    return np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(h, n), n)[:len(x) + len(h) - 1]


# Each environment is a set of knob values over the defaults (the outdoor
# range fitted to the CZ 75 recording).
ENVIRONMENTS = {
    "range": {},
    "dry": {"env.ground_kpa": 0.0, "env.near_db": -99.0, "env.echo_db": -99.0, "env.far_db": -99.0},
    "open": {  # a field: ground, little to scatter, a treeline 150 m off
        "env.ground_kpa": 200.0, "env.near_db": -26.0, "env.near_rt": 0.35, "env.near_build_ms": 15.0,
        "env.echo_delay": 0.9, "env.echo_db": -34.0, "env.echo_width_ms": 60.0, "env.far_db": -30.0, "env.far_rt": 1.5},
    "forest": {  # trunks all around: a long, dark, dense tail
        "env.ground_kpa": 100.0, "env.near_db": -6.0, "env.near_rt": 1.4, "env.near_rt_hf": 0.6,
        "env.near_build_ms": 25.0, "env.near_tilt": -4.5, "env.echo_db": -99.0, "env.far_db": -99.0},
    "urban": {  # a street: hard ground, facades, flutter between them
        "env.ground_kpa": 20000.0, "env.near_db": -8.0, "env.near_rt": 1.1, "env.near_rt_hf": 0.2,
        "env.near_build_ms": 4.0, "env.near_lo_hz": 120.0, "env.near_tilt": -1.0,
        "env.echo_delay": 0.12, "env.echo_db": -14.0, "env.echo_width_ms": 4.0, "env.far_db": -22.0, "env.far_rt": 1.6},
    "room": {  # a small concrete room: dense, fast, bright
        "env.ground_kpa": 20000.0, "env.near_db": 6.0, "env.near_rt": 0.6, "env.near_rt_hf": 0.15,
        "env.near_build_ms": 1.0, "env.near_lo_hz": 90.0, "env.near_tilt": -0.5,
        "env.echo_db": -99.0, "env.far_db": -99.0},
}
