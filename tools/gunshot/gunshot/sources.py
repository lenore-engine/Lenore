"""Pressure signals of each part, at 1 m, before propagation."""

import math
import numpy as np

from . import dsp
from .mechanism import SURFACES, Bounce

P_ATM = 101325.0
REF_CHARGE, REF_BARREL = 0.35e-3, 0.117

# Mode tables: (frequency Hz, decay time constant s, relative gain). Estimates
# from beam and shell formulas for the CZ 75's steel parts; tune by ear and
# against recordings. "damped" bodies are held and follow mech.grip_damping.
BODIES = {
    # Parts pressed against each other and held in a hand ring briefly: a
    # struck slide reads as a clack, not as a free bar.
    "slide":     (True,  [(1150, .010, .5), (2150, .014, 1.), (3400, .012, .6), (5900, .009, .7),
                          (7900, .007, .4), (11000, .005, .35), (15500, .004, .2)]),
    "frame":     (True,  [(620, .008, .3), (1350, .009, .8), (2600, .008, .7), (4100, .006, .6),
                          (6800, .005, .4), (9500, .004, .25)]),
    "barrel":    (False, [(5200, .018, 1.), (9800, .012, .4), (14300, .009, .5)]),
    "hammer":    (False, [(4400, .008, 1.), (8800, .006, .6), (13200, .005, .4)]),
    "trigger":   (True,  [(2900, .006, .6), (6100, .005, .8), (11800, .004, .4)]),
    "magazine":  (True,  [(240, .02, .4), (1800, .012, .7), (3300, .010, .6), (5200, .008, .4)]),
    "cartridge": (False, [(7200, .008, .6), (11800, .006, .4)]),
}

# Which bodies an event rings, with what weight, the contact time at 5 m/s,
# and the mass whose momentum the impact stops (kg, estimates). The sound of
# an impact scales with that momentum: a 4 g trigger bar letting go is not a
# 300 g slide hitting the frame at the same speed.
EVENTS = {
    "sear_release":   ([("hammer", .15), ("trigger", .3)], 30e-6, 0.005),
    "hammer_strike":  ([("hammer", 1.), ("slide", .4), ("frame", .3)], 40e-6, 0.025),
    # The powder pushes for the bullet's whole barrel time, ~0.5 ms: a slow
    # force that rings the low modes, not a contact.
    "fire_jolt":      ([("barrel", 1.), ("slide", .6), ("frame", .5)], 400e-6, 0.375),
    "barrel_unlock":  ([("barrel", .8), ("frame", .5)], 60e-6, 0.075),
    "hammer_cock":    ([("hammer", .6), ("frame", .4)], 40e-6, 0.025),
    "eject":          ([("frame", .4), ("slide", .3)], 40e-6, 0.012),
    "rear_stop":      ([("slide", 1.), ("frame", .9)], 70e-6, 0.30),
    "mag_bump":       ([("magazine", 1.)], 80e-6, 0.03),
    "feed_ramp":      ([("barrel", .5), ("cartridge", .6)], 50e-6, 0.012),
    "barrel_pickup":  ([("barrel", .5), ("slide", .3)], 60e-6, 0.075),
    "extractor_snap": ([("slide", .3), ("cartridge", .2)], 30e-6, 0.004),
    "battery":        ([("slide", 1.), ("barrel", .7), ("frame", .8)], 60e-6, 0.375),
    "slide_lock":     ([("slide", 1.), ("frame", .9)], 60e-6, 0.30),
    "trigger_reset":  ([("trigger", 1.), ("frame", .2)], 30e-6, 0.004),
}
REF_MOMENTUM = 0.30 * 5.0  # the slide at 5 m/s: what mech.ref_pa describes

SURFACE_MODES = {
    "concrete": [],
    "tile": [(1200, .01, .5), (2600, .008, .4)],
    "wood": [(180, .02, .6), (420, .015, .5), (950, .01, .4)],
    "dirt": [],
    "steel": [(310, .3, .5), (870, .25, .5), (1650, .2, .4), (2900, .15, .3)],
}


# Modes per kHz beyond the tabulated ones. A 4 mm steel plate the size of the
# slide has one bending mode every ~600 Hz; the table holds the strongest few,
# and with only those a struck part rings like a tuned bar instead of a clack.
DENSITY = {"slide": 1.6, "frame": 1.8, "barrel": 0.5, "hammer": 0.4, "trigger": 0.5,
           "magazine": 1.2, "cartridge": 0.3}


class Gun:
    """One physical pistol: its parts' modes, detuned once by the gun seed so
    that two pistols of the same model do not ring identically."""

    def __init__(self, gun_rng):
        self.bodies = {}
        for name, (damped, modes) in BODIES.items():
            m = np.array(modes, dtype=float)
            # Fill in the dense modes: uniform in frequency (constant modal
            # density), weaker than the tabulated ones, with the part's
            # median loss factor so they die at the same rate per cycle.
            eta = np.median(1 / (np.pi * m[:, 0] * m[:, 1]))
            count = int(DENSITY[name] * (17000 - m[0, 0]) / 1000)
            f = gun_rng.uniform(m[0, 0], 17000, count)
            g = 0.4 * np.interp(f, m[:, 0], m[:, 2]) * np.exp(0.5 * gun_rng.standard_normal(count))
            tau = 1 / (np.pi * f * eta) * np.exp(0.3 * gun_rng.standard_normal(count))
            m = np.vstack([m, np.column_stack([f, tau, g])])
            m[:, 0] *= 1.0 + 0.03 * gun_rng.standard_normal(len(m))
            m[:, 1] *= np.exp(0.2 * gun_rng.standard_normal(len(m)))
            m[:, 2] *= np.exp(0.15 * gun_rng.standard_normal(len(m)))
            self.bodies[name] = (damped, m)

    def modes(self, name, k, rng):
        damped, m = self.bodies[name]
        m = m.copy()
        m[:, 0] *= 1.0 + k["mech.detune"] * rng.standard_normal(len(m))
        if damped:
            m[:, 1] /= k["mech.grip_damping"]
        return m


def mech_event(name: str, v: float, gun: Gun, k, rng, fs) -> np.ndarray:
    """Pressure at 1 m of one mechanical impact."""
    parts, contact5, mass = EVENTS[name]
    v = max(abs(v), 0.05)
    contact = contact5 * (v / 5.0) ** -0.2  # Hertz: faster impacts are shorter
    amp = k["mech.ref_pa"] * mass * v / REF_MOMENTUM
    longest = max(gun.bodies[p][1][:, 1].max() for p, _ in parts)
    length = min(0.5, 6.0 * longest + 0.005)
    out = np.zeros(int(length * fs))
    for part, w in parts:
        sig = dsp.modal(rng, gun.modes(part, k, rng), fs, contact, length)
        out += w * sig
    peak = np.max(np.abs(out)) + 1e-20
    out /= peak
    click = dsp.contact_click(rng, fs, contact)
    out[:len(click)] += 0.5 * click
    return amp * out


def da_pull(k, gun, rng, fs) -> np.ndarray:
    """The double-action pull: the trigger bar drags the hammer back with a
    faint grind and two small catches."""
    T = k["trigger.da_pull_time"]
    n = int(T * fs)
    env = np.linspace(0.3, 1.0, n) * np.sin(np.pi * np.linspace(0, 1, n)) ** 0.3
    out = 0.003 * k["mech.ref_pa"] * env * dsp.shaped_noise(rng, n, fs, 3500, 1.0)
    for frac in (rng.uniform(.15, .3), rng.uniform(.6, .8)):
        s = mech_event("trigger_reset", .4, gun, k, rng, fs)
        dsp.add(out, int(frac * n), s)
    return out


def rail_noise(cycle, k, rng, fs, t0) -> tuple[int, np.ndarray]:
    """Steel sliding in steel rails while the slide moves."""
    if not cycle.rail_t:
        return 0, np.zeros(0)
    ts, vs = np.array(cycle.rail_t), np.array(cycle.rail_v)
    start = int((ts[0] - t0) * fs)
    n = int((ts[-1] - ts[0]) * fs) + 1
    env = np.interp(ts[0] + np.arange(n) / fs, ts, vs) / 5.0
    noise = dsp.shaped_noise(rng, n, fs, 4500, 0.8)
    level = k["mech.ref_pa"] * 10 ** (k["mech.rail_noise_db"] / 20)
    return start, level * env * noise


def blast_level(k, cos_theta, distance):
    """Peak overpressure and positive duration at the listener."""
    scale = (k["ammo.charge_mass"] / REF_CHARGE) ** (1 / 3) * (REF_BARREL / k["barrel.length"]) ** 0.3
    db = k["blast.directivity_db"] * cos_theta
    d = max(distance, 0.3)  # inside ~ the Weber radius the 1/r law stops
    peak = k["blast.peak_pa"] * scale * 10 ** (db / 20) / d
    positive = k["blast.positive_ms"] * 1e-3 * scale * d ** 0.1 * (1 + 0.25 * cos_theta)
    return peak, positive


def muzzle(k, rng, fs, t_arrive, cos_theta, distance, t0) -> list[tuple[int, np.ndarray]]:
    """Blast front, precursor, secondary flash and gas jet, at the listener."""
    peak, positive = blast_level(k, cos_theta, distance)
    rise = k["blast.rise_us"] * 1e-6
    b = k["blast.decay_b"]
    out = []
    main = dsp.friedlander(peak, positive, b, rise)
    out.append(dsp.render_analytic(main, t_arrive - t0, 8 * positive, fs))

    lead = rng.uniform(0.05e-3, 0.15e-3)
    pre = dsp.friedlander(peak * 10 ** (k["blast.precursor_db"] / 20), 0.35 * positive, 1.5, rise)
    out.append(dsp.render_analytic(pre, t_arrive - lead - t0, 3 * positive, fs))

    if rng.uniform() < k["ammo.flash_chance"]:
        fl = dsp.friedlander(peak * 10 ** (k["flash.level_db"] / 20), 2.5 * positive, 1.2, 4 * rise)
        out.append(dsp.render_analytic(fl, t_arrive + k["flash.delay_ms"] * 1e-3 - t0, 20 * positive, fs))

    out.append(dsp.render_analytic(blowdown(k, distance), t_arrive - t0, 12 * k["blowdown.time_ms"] * 1e-3, fs))

    dur = k["gas.duration_ms"] * 1e-3
    n = int(6 * dur * fs)
    tt = np.arange(n) / fs
    env = (1 - np.exp(-tt / 0.2e-3)) * np.exp(-tt / dur)
    jet_db = k["gas.level_db"] + 3.0 * cos_theta  # the jet points downrange
    jet = peak * 10 ** (jet_db / 20) * env * dsp.shaped_noise(rng, n, fs, k["gas.center_hz"], 1.3)
    out.append((int((t_arrive - t0) * fs), jet))
    return out


def blowdown(k, distance):
    """The barrel emptying: the propellant gas leaves as a volume flow
    Q(t) = V/τ (1 - e^{-t/τr}) e^{-t/τ}, a monopole p = ρ/(4πr) dQ/dt. Its
    long positive and negative lobes are the blast's body below ~1 kHz; the
    shock front alone is only a click."""
    rho, r_gas, molar = 1.2, 8.314, 0.025
    volume = k["ammo.charge_mass"] * r_gas * k["blowdown.gas_temperature"] / (P_ATM * molar)
    tau = k["blowdown.time_ms"] * 1e-3
    tr = k["blowdown.rise_ms"] * 1e-3
    q0 = volume / tau
    scale = k["blowdown.gain"] * rho / (4 * math.pi * max(distance, 0.3))

    def fn(t):
        # d/dt [(1 - e^{-t/tr}) e^{-t/tau}]
        return scale * q0 * (np.exp(-t / tr) / tr - (1 - np.exp(-t / tr)) / tau) * np.exp(-t / tau)
    return fn


def bullet_path(k, c):
    """Velocity and time along the flight with linear drag; and the
    distance at which the bullet becomes subsonic."""
    v0, kd = k["ammo.velocity"], max(k["ammo.drag"], 1e-6)

    def v(x):
        return v0 - kd * x

    def t(x):
        return -math.log(max(1 - kd * x / v0, 1e-9)) / kd

    x_sub = (v0 - c) / kd if v0 > c else 0.0
    return v, t, x_sub


def n_wave(k, rng, fs, c, t_exit, listener, t0) -> tuple[int, np.ndarray] | None:
    """Ballistic crack by Whitham's far-field formula, if the listener lies in
    the Mach cone of a supersonic stretch of the flight."""
    v, tof, x_sub = bullet_path(k, c)
    if x_sub <= 0:
        return None
    lx, ly, lz = listener
    miss = max(math.hypot(ly, lz), 0.05)
    # Find the emission point x_e: x_e = lx - miss / sqrt(M(x_e)^2 - 1).
    lo, hi = 0.0, min(lx, x_sub)
    if hi <= 0:
        return None

    def g(xe):
        M = v(xe) / c
        return xe - (lx - miss / math.sqrt(max(M * M - 1, 1e-9)))

    if g(lo) > 0 or g(hi) < 0:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
    xe = 0.5 * (lo + hi)
    M = v(xe) / c
    m2 = max(M * M - 1, 0.02)  # Whitham diverges at M = 1; hold it near there
    path = miss * M / math.sqrt(M * M - 1)
    t_arr = t_exit + tof(xe) + path / c
    d, l = k["ammo.bullet_diameter"], k["ammo.bullet_length"]
    amp = 0.53 * P_ATM * m2 ** 0.125 * d / (miss ** 0.75 * l ** 0.25)
    dur = 1.82 * M * miss ** 0.25 * d / (c * m2 ** 0.375 * l ** 0.25)
    # Whitham's slender-body far field overstates a barely supersonic bullet,
    # whose bow shock stands off and steepens slowly; fade it towards M = 1.
    amp *= min(1.0, (M * M - 1) / k["nwave.transonic_m2"])
    amp *= math.exp(0.05 * rng.standard_normal())
    return dsp.render_analytic(dsp.n_wave(amp, dur, 3e-6), t_arr - t0, dur + 1e-3, fs)


def casing(k, rng, fs, bounces: list[Bounce], roll_start, listener, c, t0):
    """The casing's ring off the ejector, its bounces and its roll."""
    name, _, ring_keep, rolls = SURFACES[int(round(k["surface.kind"]))]
    f1, q = k["case.ring_hz"], k["case.ring_q"]
    ratios = [(1.0, 1.0), (1.58, .45), (2.83, .4), (3.9, .2)]
    base = np.array([(f1 * r, q / (math.pi * f1 * r), g) for r, g in ratios])
    out = []
    surf_modes = np.array(SURFACE_MODES[name]) if SURFACE_MODES[name] else None
    for i, b in enumerate(bounces):
        m = base.copy()
        m[:, 0] *= 1 + 0.004 * rng.standard_normal(len(m))
        m[:, 1] *= ring_keep
        # Mouth hits ring the shell; base hits are duller.
        m[:, 2] *= {"mouth": 1.0, "side": 0.7, "base": 0.35}[b.end]
        contact = 25e-6 if name != "dirt" else 400e-6
        length = min(0.6, 6 * m[:, 1].max() + 0.01)
        sig = dsp.modal(rng, m, fs, contact, length)
        sig /= np.max(np.abs(sig)) + 1e-20
        click = dsp.contact_click(rng, fs, contact)
        sig[:len(click)] += (0.6 if name != "dirt" else 1.5) * click
        if surf_modes is not None:
            sm = surf_modes.copy()
            sm[:, 0] *= 1 + 0.02 * rng.standard_normal(len(sm))
            s2 = dsp.modal(rng, sm, fs, 200e-6, length)
            sig[:len(s2)] += 0.5 * s2[:len(sig)] / (np.max(np.abs(s2)) + 1e-20)
        amp = k["case.ref_pa"] * b.v / 3.0
        dist = math.dist(b.pos, listener)
        start = int((b.t + dist / c - t0) * fs)
        out.append((start, amp / max(dist, 0.1) * sig))

    roll_t = k["case.roll_time"]
    if rolls and roll_t > 0 and bounces:
        n = int(roll_t * fs)
        # Rolling over grit: a thinning train of small random impacts.
        rate = np.linspace(180, 20, n) / fs
        hits = (rng.uniform(size=n) < rate) * rng.exponential(1.0, n)
        hits *= np.linspace(1, 0, n) ** 1.5
        m = base.copy()
        m[:, 1] *= ring_keep * 0.5
        ir = dsp.modal(rng, m, fs, 30e-6, min(0.3, 6 * m[:, 1].max()))
        ir /= np.max(np.abs(ir)) + 1e-20
        size = n + len(ir)
        rolled = np.fft.irfft(np.fft.rfft(hits, size) * np.fft.rfft(ir, size), size)[:n]
        last = bounces[-1]
        dist = math.dist(last.pos, listener)
        start = int((roll_start + dist / c - t0) * fs)
        out.append((start, 0.12 * k["case.ref_pa"] / max(dist, 0.1) * rolled))
    return out


def eject_ring(k, rng, fs) -> np.ndarray:
    """The casing ringing in flight after it hits the ejector."""
    f1, q = k["case.ring_hz"], k["case.ring_q"]
    m = np.array([(f1 * r, q / (math.pi * f1 * r), g) for r, g in [(1, 1), (1.58, .45), (2.83, .4)]])
    sig = dsp.modal(rng, m, fs, 30e-6, 0.25)
    return 0.4 * k["case.ref_pa"] * sig / (np.max(np.abs(sig)) + 1e-20)
