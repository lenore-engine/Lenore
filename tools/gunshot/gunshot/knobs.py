"""Every knob of the model, its default, its per-shot spread and what it means.

A knob is addressed by a dotted path (``ammo.velocity``). Its value is drawn
once per shot as ``default + spread * N(0, 1)``, clamped to ``lo..hi``. A spread
of zero makes the knob fixed. Defaults describe a CZ 75 B with 115 gr FMJ held
firmly, heard at the shooter's right ear outdoors; figures marked "estimate" in
the spec are not measured and are there to be tuned against recordings.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Knob:
    default: float
    spread: float
    lo: float
    hi: float
    unit: str
    doc: str


K = Knob

KNOBS: dict[str, Knob] = {
    # --- environment and listener -------------------------------------------
    "air.temperature": K(20.0, 0.0, -40, 50, "°C", "air temperature; sets the speed of sound and so whether the bullet is supersonic"),
    "air.humidity": K(50.0, 0.0, 1, 100, "%", "relative humidity; with temperature, how fast the air takes the highs off every reflection"),
    "listener.distance": K(0.45, 0.0, 0.1, 500, "m", "muzzle to listener"),
    "listener.azimuth": K(140.0, 0.0, 0, 180, "deg", "0 = straight downrange, 90 = beside the line of fire, 180 = behind the muzzle"),
    "listener.height": K(1.6, 0.0, 0.1, 10, "m", "listener above the ground (casing and ground reflection)"),
    "gun.height": K(1.4, 0.0, 0.1, 10, "m", "gun above the ground; sets the casing's fall"),
    # --- trigger group ------------------------------------------------------
    "trigger.double_action": K(0.0, 0.0, 0, 1, "bool", "1 = first shot double action: the pull cocks the hammer before it falls"),
    "trigger.da_pull_time": K(0.28, 0.04, 0.1, 0.8, "s", "duration of the double-action pull"),
    "trigger.reset_delay": K(0.16, 0.04, 0.0, 1.0, "s", "shot to the trigger-reset click as the finger lets out; 0 = no reset click"),
    "hammer.travel_time": K(3.2e-3, 0.15e-3, 1e-3, 8e-3, "s", "sear release to hammer striking the firing pin"),
    "hammer.strike_velocity": K(4.5, 0.2, 1, 10, "m/s", "hammer face velocity at the strike"),
    "ignition.delay": K(0.9e-3, 0.2e-3, 0.2e-3, 3e-3, "s", "firing pin strike to the bullet starting to move (primer and powder)"),
    # --- ammunition ---------------------------------------------------------
    "ammo.velocity": K(350.0, 8.0, 250, 450, "m/s", "muzzle velocity; 115 gr from a 4.6 in barrel measures ~345-355"),
    "ammo.bullet_mass": K(7.45e-3, 0.0, 4e-3, 10e-3, "kg", "bullet mass (115 gr = 7.45 g, 124 gr = 8.0 g, 147 gr = 9.5 g)"),
    "ammo.bullet_length": K(15.5e-3, 0.0, 10e-3, 20e-3, "m", "bullet length (Whitham N-wave)"),
    "ammo.bullet_diameter": K(9.02e-3, 0.0, 8e-3, 10e-3, "m", "bullet diameter"),
    "ammo.charge_mass": K(0.35e-3, 0.01e-3, 0.1e-3, 0.6e-3, "kg", "propellant mass; drives blast energy and gas momentum"),
    "ammo.drag": K(0.85, 0.0, 0, 3, "(m/s)/m", "linear velocity loss along the flight"),
    "ammo.flash_chance": K(0.25, 0.0, 0, 1, "p", "probability of a secondary flash behind the muzzle"),
    "barrel.length": K(0.117, 0.0, 0.05, 0.3, "m", "barrel length"),
    # --- muzzle blast -------------------------------------------------------
    "blast.peak_pa": K(1100.0, 80.0, 10, 20000, "Pa", "peak overpressure at 1 m, 90° off the bore, for the reference charge"),
    "blast.directivity_db": K(7.0, 0.0, 0, 15, "dB", "level swing with angle: +d on the bore axis, -d behind the muzzle"),
    "blast.positive_ms": K(0.15, 0.02, 0.03, 3.0, "ms", "positive-phase duration at 1 m, 90° (Kinney-Graham 0.13, fit to a CZ 75 recording)"),
    "blast.decay_b": K(1.8, 0.15, 0.5, 5, "", "Friedlander decay constant: larger = sharper, less low end"),
    "blast.rise_us": K(12.0, 3.0, 1, 100, "µs", "rise time of the front"),
    "blast.precursor_db": K(-22.0, 2.0, -60, 0, "dB", "precursor (air pushed out ahead of the bullet) relative to the main blast"),
    "blowdown.gas_temperature": K(450.0, 40.0, 250, 3000, "K", "gas temperature once expanded to ambient (~1300 K at the muzzle, adiabatic from ~400 bar); sets the volume, ~0.5 L"),
    "blowdown.time_ms": K(0.2, 0.03, 0.03, 10, "ms", "barrel emptying time constant (choked flow empties the bore in ~0.1-0.3 ms); longer = duller, a thud"),
    "blowdown.rise_ms": K(0.07, 0.01, 0.01, 2, "ms", "how fast the gas flow starts"),
    "blowdown.gain": K(0.15, 0.0, 0, 10, "×", "scale on the blowdown monopole; the free-field monopole overstates a directed, nonlinear jet this close (set so the blast peaks ~157 dB at the ear)"),
    "gas.level_db": K(-24.0, 2.0, -60, 0, "dB", "turbulent jet noise after the front, relative to the blast peak"),
    "gas.duration_ms": K(9.0, 2.0, 1, 50, "ms", "jet noise decay time"),
    "gas.center_hz": K(2500.0, 400.0, 300, 12000, "Hz", "jet noise spectral peak"),
    "flash.level_db": K(-12.0, 3.0, -40, 0, "dB", "secondary flash relative to the blast"),
    "flash.delay_ms": K(0.8, 0.3, 0.1, 5, "ms", "secondary flash delay after the blast"),
    "nwave.transonic_m2": K(0.25, 0.0, 0.01, 2, "", "M²-1 below which the crack fades linearly (estimate: Whitham overstates near M = 1)"),
    # --- slide and recoil (all estimates) -----------------------------------
    "slide.mass": K(0.30, 0.0, 0.1, 0.6, "kg", "slide mass (all-steel CZ 75)"),
    "slide.barrel_mass": K(0.075, 0.0, 0.03, 0.2, "kg", "barrel mass"),
    "slide.frame_mass": K(1.6, 0.15, 0.4, 5, "kg", "frame plus the effective mass of the hands; low = limp wrist"),
    "slide.stroke": K(33e-3, 0.0, 25e-3, 45e-3, "m", "slide travel to the frame stop"),
    "slide.unlock_at": K(3.5e-3, 0.0, 1e-3, 8e-3, "m", "travel at which the barrel tilts down and stops"),
    "spring.preload": K(40.0, 2.0, 5, 120, "N", "recoil spring force in battery"),
    "spring.rate": K(1050.0, 30.0, 100, 5000, "N/m", "recoil spring rate"),
    "hammer.cock_force": K(35.0, 3.0, 0, 120, "N", "resistance of the mainspring while the slide cocks the hammer"),
    "feed.strip_force": K(14.0, 3.0, 0, 60, "N", "drag of stripping the next round from the magazine"),
    "rails.friction": K(4.0, 0.5, 0, 30, "N", "slide-in-frame rail friction"),
    "impact.rear_restitution": K(0.30, 0.04, 0, 0.9, "", "slide-to-frame stop restitution"),
    "impact.battery_restitution": K(0.12, 0.03, 0, 0.9, "", "slide into battery restitution"),
    "magazine.rounds": K(15.0, 0.0, 0, 16, "", "rounds left in the magazine after this shot is chambered; 0 locks the slide back"),
    # --- mechanical sound ---------------------------------------------------
    "mech.ref_pa": K(8.0, 0.0, 0.01, 1000, "Pa", "peak pressure at 1 m of the slide (0.3 kg) hitting steel at 5 m/s; other impacts scale with their momentum"),
    "mech.grip_damping": K(1.0, 0.1, 0.2, 5, "×", "multiplies the decay rate of frame and slide modes (a firm grip damps them)"),
    "mech.detune": K(0.012, 0.0, 0, 0.1, "", "per-shot random detune of each mode (relative)"),
    "mech.rail_noise_db": K(-30.0, 2.0, -80, 0, "dB", "rail sliding noise relative to the impacts"),
    # --- casing -------------------------------------------------------------
    "case.eject_speed": K(3.6, 0.5, 1, 10, "m/s", "casing speed off the ejector"),
    "case.eject_elevation": K(40.0, 10.0, -20, 85, "deg", "ejection angle above horizontal"),
    "case.eject_azimuth": K(100.0, 12.0, 0, 180, "deg", "ejection direction from the bore, to the right"),
    "case.ring_hz": K(10850.0, 60.0, 3000, 20000, "Hz", "first ovalling mode of the 9×19 brass case (10.84-10.87 kHz in a CZ 75 recording)"),
    "case.ring_q": K(1300.0, 150.0, 20, 2000, "", "quality factor of the case modes in free air"),
    "case.ref_pa": K(0.6, 0.0, 0.001, 100, "Pa", "peak at 1 m of a 3 m/s casing impact on a hard floor"),
    "case.roll_time": K(0.6, 0.25, 0, 3, "s", "how long the casing rolls after it stops bouncing (hard surfaces)"),
    "surface.kind": K(0.0, 0.0, 0, 4, "enum", "0 concrete, 1 tile, 2 wood, 3 dirt, 4 steel plate"),
    # --- environment (fitted to a CZ 75 recorded outdoors at ~0.5 m) --------
    "env.ground_kpa": K(300.0, 0.0, 0, 1e5, "kPa·s/m²", "ground flow resistivity: ~200 grass, ~2000 packed dirt, 2e4+ asphalt or concrete; 0 = no ground"),
    "env.near_db": K(-6.0, 0.0, -99, 20, "dB", "scattered field near the shooter (trees, berms, walls): energy at 1 kHz re the free-field source at 1 m"),
    "env.near_rt": K(0.6, 0.0, 0.05, 6, "s", "its RT60 at 1 kHz before the air takes its share"),
    "env.near_rt_hf": K(0.3, 0.0, 0, 2, "", "RT60 shortens as (1 kHz / f) to this power above 1 kHz (foliage, soft ground)"),
    "env.near_build_ms": K(5.0, 0.0, 0.1, 300, "ms", "how long the scattered field takes to build up"),
    "env.near_lo_hz": K(250.0, 0.0, 10, 2000, "Hz", "below this the scatterers are small against the wavelength: the field falls 12 dB/oct"),
    "env.near_tilt": K(-3.0, 0.0, -12, 6, "dB/oct", "spectral tilt of the scattered field above 1 kHz"),
    "env.echo_delay": K(0.625, 0.0, 0, 3, "s", "delay of the strongest distant return (a treeline or hill at c·t/2); 0 = none"),
    "env.echo_db": K(-30.0, 0.0, -99, 10, "dB", "its energy re the free-field source at 1 m"),
    "env.echo_width_ms": K(35.0, 0.0, 1, 300, "ms", "its spread: a face of scattered trees returns a burst, a wall a click"),
    "env.far_db": K(-26.0, 0.0, -99, 10, "dB", "terrain rolling the shot on after the echo, energy re the source at 1 m"),
    "env.far_rt": K(2.0, 0.0, 0.1, 10, "s", "its RT60"),
    # --- receiver: what records the shot (fitted to a phone) ----------------
    "rec.clip_db": K(118.0, 0.0, 90, 200, "dB SPL", "microphone overload: the pressure is soft-clipped about here"),
    "rec.highpass_hz": K(150.0, 0.0, 0, 1000, "Hz", "Butterworth high-pass of the microphone and its pipeline"),
    "rec.highpass_order": K(1.0, 0.0, 1, 4, "×2", "its order in 2nd-order sections"),
    "rec.lowpass_hz": K(16000.0, 0.0, 1000, 24000, "Hz", "the codec's band limit (a brick wall)"),
    "rec.agc_threshold_db": K(100.0, 0.0, 40, 160, "dB SPL", "level the gain control holds the signal to"),
    "rec.agc_ratio": K(10.0, 0.0, 1, 100, ":1", "compression above the threshold; 1 = no gain control"),
    "rec.agc_window_ms": K(3.0, 0.0, 0.1, 100, "ms", "RMS window of its level detector"),
    "rec.agc_release": K(500.0, 0.0, 1, 1e4, "dB/s", "how fast its gain climbs back after the blast"),
    "rec.agc_lookahead_ms": K(1.0, 0.0, 0, 10, "ms", "how early it sees what is coming: with none, the blast's first spike gets through"),
    # --- mix ----------------------------------------------------------------
    "mix.blast_db": K(0.0, 0.0, -80, 40, "dB", "artistic gain on blast + gas + flash"),
    "mix.nwave_db": K(0.0, 0.0, -80, 40, "dB", "artistic gain on the ballistic crack"),
    "mix.mech_db": K(0.0, 0.0, -80, 60, "dB", "artistic gain on the mechanism"),
    "mix.case_db": K(0.0, 0.0, -80, 60, "dB", "artistic gain on the casing"),
    "mix.full_scale_pa": K(2000.0, 0.0, 1, 1e5, "Pa", "pressure written as 1.0 in the WAV"),
}


def draw(rng, overrides: dict[str, float], spreads: dict[str, float]) -> dict[str, float]:
    """One shot's values: overridden defaults with their spread applied."""
    values = {}
    for path, knob in KNOBS.items():
        base = overrides.get(path, knob.default)
        spread = spreads.get(path, knob.spread)
        value = base + spread * rng.standard_normal() if spread > 0 else base
        values[path] = min(max(value, knob.lo), knob.hi)
    return values


def describe() -> str:
    rows = []
    for path, k in KNOBS.items():
        spread = f"±{k.spread:g}" if k.spread else ""
        rows.append(f"{path:28} {k.default:<10g}{spread:<9}{k.unit:<9}{k.doc}")
    return "\n".join(rows)
