"""Signal building blocks, numpy only."""

import numpy as np

OVERSAMPLE = 8


def lowpass_decimate(x: np.ndarray, factor: int) -> np.ndarray:
    """Band-limit to 0.45 of the target rate and keep every ``factor``-th sample."""
    pad = 256 * factor
    n = len(x) + 2 * pad
    spec = np.fft.rfft(np.concatenate([np.zeros(pad), x, np.zeros(pad)]))
    f = np.fft.rfftfreq(n, 1.0)  # cycles per input sample
    edge, width = 0.45 / factor, 0.05 / factor
    w = np.clip((edge + width - f) / width, 0.0, 1.0)
    spec *= 0.5 - 0.5 * np.cos(np.pi * w)
    y = np.fft.irfft(spec, n)[pad:pad + len(x)]
    return y[::factor].copy()


def render_analytic(fn, t0: float, length: float, fs: float) -> tuple[int, np.ndarray]:
    """Sample ``fn(t)`` (t in seconds from its own onset) placed at absolute
    time ``t0``, band-limited. Returns the start index at ``fs`` and the
    samples; the onset lands between samples exactly."""
    start = int(np.floor(t0 * fs)) - 32
    n = int(np.ceil(length * fs)) + 64
    hi = fs * OVERSAMPLE
    t = (start + np.arange(n * OVERSAMPLE) / OVERSAMPLE) / fs - t0
    y = np.where(t >= 0, fn(np.maximum(t, 0.0)), 0.0)
    return start, lowpass_decimate(y, OVERSAMPLE)


def smooth_step(t: np.ndarray, rise: float) -> np.ndarray:
    """Front of finite rise time: 0 at t=0, ~1 after ``rise``."""
    return 1.0 - np.exp(-t / max(rise, 1e-9) * 3.0)


def friedlander(peak: float, positive: float, b: float, rise: float):
    """Friedlander blast wave p(t) = P (1 - t/T) e^{-b t / T} with a finite front."""
    def fn(t):
        return peak * smooth_step(t, rise) * (1.0 - t / positive) * np.exp(-b * t / positive)
    return fn


def n_wave(peak: float, duration: float, rise: float):
    """Ballistic N-wave: jump to +P, linear fall to -P over T, jump back."""
    def fn(t):
        ramp = peak * (1.0 - 2.0 * t / duration)
        front = smooth_step(t, rise)
        back = 1.0 - smooth_step(np.maximum(t - duration, 0.0), rise)
        return np.where(t <= duration, ramp * front, -peak * back)
    return fn


def shaped_noise(rng, n: int, fs: float, center: float, octaves: float) -> np.ndarray:
    """Unit-RMS noise with a log-Gaussian band around ``center``."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    with np.errstate(divide="ignore"):
        lf = np.log2(np.maximum(f, 1.0) / center)
    spec *= np.exp(-0.5 * (lf / octaves) ** 2)
    y = np.fft.irfft(spec, n)
    return y / (np.sqrt(np.mean(y * y)) + 1e-20)


def half_sine_weight(f: np.ndarray, contact: float) -> np.ndarray:
    """Magnitude spectrum of a half-sine force pulse of length ``contact``,
    normalised to 1 at DC. A shorter contact rings higher modes."""
    x = 2.0 * f * contact
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.abs(np.cos(np.pi * x / 2.0) / (1.0 - x * x))
    return np.where(np.abs(1.0 - x * x) < 1e-6, np.pi / 4.0, w)


def modal(rng, modes: np.ndarray, fs: float, contact: float, length: float) -> np.ndarray:
    """Sum of exponentially decaying sinusoids.

    ``modes`` rows are (frequency Hz, decay time constant s, gain). Phases are
    random so that repeated impacts do not cancel identically.
    """
    n = int(length * fs)
    t = np.arange(n) / fs
    out = np.zeros(n)
    w = half_sine_weight(modes[:, 0], contact)
    for (f, tau, g), wi in zip(modes, w):
        if f >= 0.48 * fs:
            continue
        phase = rng.uniform(0, 2 * np.pi)
        out += g * wi * np.exp(-t / tau) * np.sin(2 * np.pi * f * t + phase)
    # A struck mode starts from zero displacement: soften the first samples.
    attack = min(n, max(2, int(contact * fs)))
    out[:attack] *= np.sin(0.5 * np.pi * np.arange(attack) / attack)
    return out


def contact_click(rng, fs: float, contact: float) -> np.ndarray:
    """The broadband snap of two hard surfaces meeting, ~one contact long."""
    n = max(8, int(4 * contact * fs) + 8)
    t = np.arange(n) / fs
    y = rng.standard_normal(n) * np.exp(-t / max(contact, 1.0 / fs))
    y -= np.convolve(y, np.ones(4) / 4, mode="same")  # crude high-pass
    return y / (np.max(np.abs(y)) + 1e-20)


def add(buf: np.ndarray, start: int, sig: np.ndarray, gain: float = 1.0) -> None:
    if start >= len(buf) or start + len(sig) <= 0:
        return
    a, b = max(start, 0), min(start + len(sig), len(buf))
    buf[a:b] += gain * sig[a - start:b - start]
