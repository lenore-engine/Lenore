"""What records or hears the shot. Nonlinear, numpy only.

At half a metre a 9 mm blast peaks near 157 dB SPL; no phone or camera
records that. Its microphone overloads, its gain control drops the gain to
hold the blast and lets it climb back through the tail, and its codec cuts
above 16 kHz. Almost every gunshot anyone has heard on a recording went
through that, and a dry, faithful rendering sounds like a click next to it.
The defaults are fitted to a phone recording of a CZ 75 outdoors.
"""

import math

import numpy as np

P0 = 2e-5


def biquad_response(F, fs, kind, f0, q):
    """Frequency response of a bilinear-transform Butterworth section,
    evaluated on the FFT grid: multiplying a zero-padded spectrum by it is the
    causal IIR filter."""
    w0 = math.tan(math.pi * min(f0, 0.49 * fs) / fs)
    norm = 1 / (1 + w0 / q + w0 * w0)
    if kind == "high":
        b = (norm, -2 * norm, norm)
    else:
        b = (w0 * w0 * norm, 2 * w0 * w0 * norm, w0 * w0 * norm)
    a = (1.0, 2 * (w0 * w0 - 1) * norm, (1 - w0 / q + w0 * w0) * norm)
    z1 = np.exp(-2j * np.pi * F / fs)
    return (b[0] + b[1] * z1 + b[2] * z1 * z1) / (a[0] + a[1] * z1 + a[2] * z1 * z1)


def butterworth(F, fs, kind, f0, order):
    """Cascade of biquads: an even-order Butterworth."""
    H = np.ones(len(F), complex)
    for i in range(order // 2):
        q = 1 / (2 * math.cos(math.pi * (2 * i + 1) / (2 * order)))
        H *= biquad_response(F, fs, kind, f0, q)
    return H


def band_limit(x, fs, hp, hp_order, lp):
    """Causal Butterworth high-pass; the codec's cut is a brick wall, as an
    MP3 or AAC encoder's filter bank is: a 0.5 kHz transition, zero phase."""
    n = 1 << int(math.ceil(math.log2(len(x) + fs * 0.1)))
    F = np.fft.rfftfreq(n, 1 / fs)
    H = np.ones(len(F), complex)
    if hp > 0:
        H *= butterworth(F, fs, "high", hp, hp_order)
    if 0 < lp < 0.5 * fs:
        H *= np.clip((lp + 250 - F) / 500, 0, 1)
    return np.fft.irfft(np.fft.rfft(x, n) * H, n)[:len(x)]


def agc_gain(x, fs, threshold_db, ratio, window_ms, release_db_s, lookahead_ms):
    """Gain of an automatic gain control: RMS level over ``window_ms``,
    instant attack ``lookahead_ms`` early (a recorder's limiter delays the
    signal to catch its own transients), a release that climbs
    ``release_db_s`` dB per second, and ``ratio``:1 compression of whatever
    lies above the threshold."""
    n = max(1, int(window_ms * 1e-3 * fs))
    c = np.concatenate([[0.0], np.cumsum(x * x)])
    ms = (c[n:] - c[:-n]) / n
    ms = np.concatenate([c[1:n] / np.arange(1, n), ms])
    level = 10 * np.log10(ms / (P0 * P0) + 1e-12)
    t = np.arange(len(x)) / fs
    # Peak hold falling at release_db_s: max over s <= t of level(s) - R (t - s).
    held = np.maximum.accumulate(level + release_db_s * t) - release_db_s * t
    over = np.maximum(held - threshold_db, 0.0)
    ahead = int(lookahead_ms * 1e-3 * fs)
    if ahead:
        over = np.concatenate([over[ahead:], np.full(ahead, over[-1])])
    return 10 ** (-over * (1 - 1 / ratio) / 20)


def process(x_pa: np.ndarray, fs, k, gain=None):
    """Pressure at the microphone (Pa) to the recorder's output, in Pa of the
    threshold's level. ``gain`` reuses another signal's gain curve (stems
    ride the mix's gain); returns the output and the gain curve."""
    clip = P0 * 10 ** (k["rec.clip_db"] / 20)
    y = clip * np.tanh(x_pa / clip)
    y = band_limit(y, fs, k["rec.highpass_hz"], 2 * int(round(k["rec.highpass_order"])), k["rec.lowpass_hz"])
    if gain is None:
        gain = agc_gain(y, fs, k["rec.agc_threshold_db"], k["rec.agc_ratio"],
                        k["rec.agc_window_ms"], k["rec.agc_release"], k["rec.agc_lookahead_ms"])
    return y * gain, gain


RECEIVERS = {
    "phone": {},
    # A listener's ear as a game mix wants it: no codec, a gentler gain
    # control, the blast kept a little above its tail.
    "ear": {"rec.clip_db": 150.0, "rec.highpass_hz": 25.0, "rec.lowpass_hz": 20000.0,
            "rec.agc_ratio": 4.0, "rec.agc_window_ms": 5.0, "rec.agc_release": 250.0},
}
