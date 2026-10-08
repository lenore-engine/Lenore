"""The firing cycle as mechanics: when each part hits what, and how hard.

Time zero is the sear releasing the hammer. Positions along the slide's travel
are measured rearward from battery, relative to the frame.
"""

from dataclasses import dataclass, field
import math

G = 9.81

# Travel (m) at which things happen along the stroke; CZ 75 estimates.
COCK_START, COCK_FULL = 6e-3, 19e-3
EJECT_AT = 26e-3
STRIP_FROM, STRIP_TO = 26e-3, 8e-3
MAG_BUMP_AT = 21e-3
FEED_RAMP_AT = 12e-3
EXTRACTOR_AT = 0.8e-3
SLIDE_STOP_GAP = 4e-3


def speed_of_sound(celsius: float) -> float:
    return 331.3 * math.sqrt(1.0 + celsius / 273.15)


@dataclass
class Event:
    t: float
    name: str
    v: float  # impact speed, m/s


@dataclass
class Cycle:
    c: float
    t_exit: float
    bullet_velocity: float
    events: list[Event] = field(default_factory=list)
    rail_t: list[float] = field(default_factory=list)
    rail_v: list[float] = field(default_factory=list)
    eject_t: float | None = None
    notes: list[str] = field(default_factory=list)


def run(k: dict[str, float]) -> Cycle:
    c = speed_of_sound(k["air.temperature"])
    v_b = k["ammo.velocity"]
    t_strike = k["hammer.travel_time"]
    t_start = t_strike + k["ignition.delay"]
    barrel_time = 2.0 * k["barrel.length"] / v_b  # uniform acceleration
    t_exit = t_start + barrel_time
    cy = Cycle(c=c, t_exit=t_exit, bullet_velocity=v_b)
    ev = cy.events

    if k["trigger.double_action"] >= 0.5:
        ev.append(Event(-k["trigger.da_pull_time"], "da_pull", 1.0))
    ev.append(Event(0.0, "sear_release", 1.0))
    ev.append(Event(t_strike, "hammer_strike", k["hammer.strike_velocity"]))

    # Free recoil momentum: bullet plus propellant gas leaving at ~1.6 v.
    p = k["ammo.bullet_mass"] * v_b + k["ammo.charge_mass"] * 1.6 * v_b
    m_s = k["slide.mass"] + k["slide.barrel_mass"]
    m_f = k["slide.frame_mass"]
    vs, vf = p / m_s, 0.0  # rearward positive, frame starts at rest
    xs = xf = 0.0
    ev.append(Event(t_exit, "fire_jolt", vs))

    stroke = k["slide.stroke"]
    unlock = k["slide.unlock_at"]
    rounds = int(round(k["magazine.rounds"]))
    k_hand, c_hand = 3000.0, 60.0
    dt = 5e-6
    t = t_start + 0.5 * barrel_time
    barrel_on = True
    locked_back = False
    reached = set()
    prev_x = 0.0
    rail_every, rail_n = 10, 0

    def spring(x):
        return k["spring.preload"] + k["spring.rate"] * x

    while t < 0.25:
        x, vrel = xs - xf, vs - vf
        f = -spring(x)
        if vrel > 0 and COCK_START < x < COCK_FULL:
            f -= k["hammer.cock_force"]
        if vrel < 0 and rounds > 0 and STRIP_TO < x < STRIP_FROM:
            f += k["feed.strip_force"]
        if abs(vrel) > 1e-4:
            f -= math.copysign(k["rails.friction"], vrel)
        a_s = f / m_s
        a_f = (-f - k_hand * xf - c_hand * vf) / m_f
        vs += a_s * dt
        vf += a_f * dt
        xs += vs * dt
        xf += vf * dt
        t += dt
        x, vrel = xs - xf, vs - vf

        rail_n += 1
        if rail_n % rail_every == 0:
            cy.rail_t.append(t)
            cy.rail_v.append(abs(vrel))

        def crossed(at, going_back):
            return (prev_x < at <= x) if going_back else (prev_x > at >= x)

        if barrel_on and crossed(unlock, True):
            # The barrel tilts out of the slide and stops on the frame.
            ev.append(Event(t, "barrel_unlock", vrel))
            mb = k["slide.barrel_mass"]
            vf += mb * vrel / (m_f + mb)
            m_f += mb
            m_s -= mb
            barrel_on = False
        if "cock" not in reached and crossed(COCK_FULL, True):
            reached.add("cock")
            ev.append(Event(t, "hammer_cock", 0.35 * vrel))
        if "eject" not in reached and crossed(EJECT_AT, True):
            reached.add("eject")
            cy.eject_t = t
            ev.append(Event(t, "eject", vrel))
        if x >= stroke and vrel > 0:
            ev.append(Event(t, "rear_stop", vrel))
            xs = xf + stroke
            e = k["impact.rear_restitution"]
            vcm = (m_s * vs + m_f * vf) / (m_s + m_f)
            vs = vcm - e * vrel * m_f / (m_s + m_f)
            vf = vcm + e * vrel * m_s / (m_s + m_f)
        elif vrel < 0:
            if rounds > 0:
                if "bump" not in reached and crossed(MAG_BUMP_AT, False):
                    reached.add("bump")
                    ev.append(Event(t, "mag_bump", 1.5))
                if "ramp" not in reached and crossed(FEED_RAMP_AT, False):
                    reached.add("ramp")
                    ev.append(Event(t, "feed_ramp", 0.5 * -vrel))
                if "extract" not in reached and crossed(EXTRACTOR_AT, False):
                    reached.add("extract")
                    ev.append(Event(t, "extractor_snap", 1.2))
            elif crossed(stroke - SLIDE_STOP_GAP, False):
                ev.append(Event(t, "slide_lock", -vrel))
                locked_back = True
                break
            if not barrel_on and crossed(unlock, False):
                ev.append(Event(t, "barrel_pickup", -vrel * 0.4))
                mb = k["slide.barrel_mass"]
                vs = (m_s * vs + mb * vf) / (m_s + mb)
                m_s += mb
                m_f -= mb
                barrel_on = True
        if x <= 0.0 and vrel < 0:
            ev.append(Event(t, "battery", -vrel))
            xs = xf
            e = k["impact.battery_restitution"]
            vcm = (m_s * vs + m_f * vf) / (m_s + m_f)
            vs = vcm - e * vrel * m_f / (m_s + m_f)
            vf = vcm + e * vrel * m_s / (m_s + m_f)
            if -vrel < 0.15:
                break
        prev_x = xs - xf

    if "eject" not in reached:
        cy.notes.append("slide short-stroked: casing not ejected")
    if not locked_back and not any(e.name == "rear_stop" for e in ev):
        cy.notes.append("slide never reached the frame stop")
    if locked_back:
        cy.notes.append("magazine empty: slide locked back")
    reset = k["trigger.reset_delay"]
    if reset > 0:
        ev.append(Event(t_exit + reset, "trigger_reset", 1.0))
    ev.sort(key=lambda e: e.t)
    return cy


@dataclass
class Bounce:
    t: float
    pos: tuple[float, float, float]
    v: float      # normal impact speed
    end: str      # "mouth", "base" or "side"


SURFACES = {
    # restitution, ring damping on the case (×decay time), rolls
    0: ("concrete", 0.28, 1.0, True),
    1: ("tile", 0.32, 1.0, True),
    2: ("wood", 0.22, 0.55, True),
    3: ("dirt", 0.05, 0.10, False),
    4: ("steel", 0.35, 1.0, True),
}


def casing_flight(k, rng, t_eject: float, origin) -> tuple[list[Bounce], float]:
    """Ballistic flight of the casing and its bounces. Returns the bounces and
    the time it starts rolling."""
    _, e, _, _ = SURFACES[int(round(k["surface.kind"]))]
    sp = k["case.eject_speed"]
    el = math.radians(k["case.eject_elevation"])
    az = math.radians(k["case.eject_azimuth"])
    vx = sp * math.cos(el) * math.cos(az)
    vy = -sp * math.cos(el) * math.sin(az)  # right of the bore is -y
    vz = sp * math.sin(el)
    x, y, z = origin
    t = t_eject
    bounces = []
    for i in range(12):
        # z + vz T - g T^2 / 2 = 0
        T = (vz + math.sqrt(vz * vz + 2 * G * z)) / G
        x, y = x + vx * T, y + vy * T
        v_imp = vz - G * T
        t += T
        end = rng.choice(["mouth", "base", "side", "side"])
        bounces.append(Bounce(t, (x, y, 0.0), -v_imp, end))
        vz = -v_imp * e * rng.uniform(0.7, 1.15)
        # Tumbling turns some vertical energy sideways and back.
        h = rng.uniform(0.4, 0.8)
        vx, vy = vx * h + rng.normal(0, 0.2), vy * h + rng.normal(0, 0.2)
        z = 0.0
        if vz < 0.25:
            break
    return bounces, t
