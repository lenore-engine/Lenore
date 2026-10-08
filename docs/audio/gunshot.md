# Procedural pistol shot: CZ 75 B, 9×19

The source model for one shot of the game's pistol, part by part. Propagation
(distance, air absorption, reflections, occlusion) is the engine's separate
concern; this model ends where pressure leaves each source, and says where
each source is so the propagation stage can place it.

`tools/gunshot` renders this model to WAV with every parameter exposed as a
knob. It is a listening tool for tuning the model before the engine carries
it; it is not engine code.

## What a shot is made of

A shot is not one sound. It is a chain of mechanical events over ~25 ms, one
blast that masks most of them at the shooter's ear, an optional ballistic
crack downrange, and brass that keeps sounding for a second or two after.

| Layer | Source | Character | Level at the shooter's ear |
|---|---|---|---|
| precursor | air pushed out ahead of the bullet | weak pulse, 0.05–0.15 ms before the blast | ~ −20 dB re blast |
| muzzle blast | propellant gas leaving the muzzle | Friedlander pulse, 0.3–1 ms positive phase | ~ 155–160 dB peak |
| gas jet | turbulent jet behind the front | broadband noise, decays in ~10 ms | ~ −25 dB re blast |
| secondary flash | unburnt gas igniting in air | second, softer, slower pulse ~1 ms later | some shots only |
| ballistic crack | supersonic bullet (N-wave) | 0.1–0.3 ms N, only inside the Mach cone | none behind the muzzle |
| mechanism | hammer, slide, barrel, magazine, trigger | modal rings of steel parts | ~ −30 to −45 dB re blast |
| brass | casing off the ejector, bouncing, rolling | high, long-ringing tinks | ~ −60 to −70 dB re blast |

The large gaps are physical. A game mix closes them (`presets/game.json`); the
model keeps them physical and the mix gains separate.

## Timeline

Time zero is the sear releasing the hammer. Figures are the model's defaults
for the CZ 75 B; those not marked as measured are estimates to be tuned.

| t (ms) | Event | Model |
|---|---|---|
| −300…0 | double-action pull (first shot only) | grind + two catches |
| 0 | sear releases the hammer | small click |
| 3.2 | hammer strikes the firing pin | hammer, slide, frame modes |
| 4.1 | bullet starts (primer + powder delay ~0.9 ms) | — |
| 4.8 | bullet exits; blast; recoil jolt rings the barrel | barrel time = 2L/v |
| +0.1 | barrel unlocks (3.5 mm of slide travel), drops onto the frame | |
| +2.4 | hammer reaches full cock (sear click) | |
| +3.5 | casing hits the ejector, leaves ringing | |
| +4.7 | slide hits the frame stop | loudest mechanical event |
| +10 | next round leaves the magazine lips; follower bumps | magazine |
| +13 | round nose on the feed ramp | |
| +16 | slide picks up the barrel | |
| +17 | extractor snaps over the rim, slide into battery | second loudest |
| +20…25 | battery bounce | small |
| +0.9…1.2 s | casing lands, bounces, rolls | surface dependent |
| +160 ms | trigger reset as the finger lets out | |

With an empty magazine the follower lifts the slide stop and the slide locks
back ~2 ms after hitting the frame stop. Nothing feeds and nothing returns to
battery.

The times after the blast are not tabulated in the model. They come from
integrating the slide and frame as two masses joined by the recoil spring:

- initial slide velocity from free-recoil momentum,
  `p = m_bullet·v + m_charge·1.6·v`, over slide + barrel mass (≈7.5 m/s);
- spring `F = F0 + k·x`, mainspring resistance while cocking the hammer,
  stripping drag while feeding, rail friction;
- the frame is the gun's frame plus the effective mass of the hands, on a weak
  spring to the arm. A low frame mass is a limp wrist: the frame recoils with
  the slide and the cycle slows (battery at +29 ms instead of +17 ms);
- the barrel leaves the slide's mass at unlock and rejoins it at pickup;
- rear stop and battery are collisions with restitution.

So hot ammunition, a weaker spring or a loose grip change every later time and
every impact's strength consistently, without per-event tuning. A slide that
fails to reach the ejector leaves the casing in the gun (logged as a note).

## Muzzle blast

The blast at the listener is a Friedlander wave with a finite front:

```
p(t) = P · (1 − t/T₊) · exp(−b·t/T₊)        t ≥ 0
```

- `P` at 1 m and 90° off the bore defaults to 1100 Pa (155 dB). The one
  measured figure found for a 9 mm pistol near the shooter's ear is 154 dB
  mean, 0.3 ms duration (Ylikoski); the 157–167 dB range quoted for 9 mm in
  suppressor literature has no stated geometry.
- Distance: `1/r`, held at 0.3 m inside the Weber radius (~0.4 m for a
  handgun), where the blast has not yet become an acoustic wave.
- Directivity: `+d·cos θ` dB with `d = 7`, a 14 dB swing from the bore axis to
  directly behind. Measured pistol directivity exceeds 20 dB between the axis
  and the side or rear in quasi-anechoic recordings (Routh and Maher); 14 dB
  is a conservative default.
- `T₊` grows with charge energy (`E^{1/3}`), slowly with distance (`r^{0.1}`)
  and toward the bore axis; the Weber model puts a larger source radius, and
  so a lower spectrum, in front of the muzzle.
- Charge and barrel: `P, T₊ ∝ (charge)^{1/3} · (L_ref/L)^{0.3}`. A shorter
  barrel leaves more pressure at the muzzle.

The Friedlander front alone is a click: at 1 m its positive phase is under a
millisecond and almost nothing of it lies below 500 Hz. The body of the blast
is the barrel emptying. The propellant gas, ~1.5 L at ambient pressure for
0.35 g of powder at ~1300 K, leaves as a volume flow
`Q(t) = (V/τ)·(1 − e^{−t/τ_r})·e^{−t/τ}` with `τ ≈ 1.2 ms`, `τ_r ≈ 0.4 ms`,
and radiates as a monopole, `p = ρ/(4πr) · dQ/dt`: a positive lobe and a
longer negative one, several hundred pascals at 0.5 m, centred near 150 Hz.
Front and blowdown together put about a quarter of the blast's energy below
500 Hz and stretch it to ~2.5 ms, inside the 3–5 ms Maher reports.

The dry blast near the gun is still a sharp crack with little low end; most of the
"boom" a listener hears is the ground reflection and the space. Judge the
model with `--room ground` or `--room room`, which stand in for propagation.

## Ballistic crack

Whitham's far-field N-wave for a slender projectile at miss distance `b`:

```
P_N = 0.53 · P_atm · (M² − 1)^{1/8} · d / (b^{3/4} · l^{1/4})
T_N = 1.82 · M · b^{1/4} · d / (c · (M² − 1)^{3/8} · l^{1/4})
```

The bullet slows linearly (`ammo.drag`, ~0.85 m/s per m for 115 gr). The
emission point `x_e` on the trajectory solves
`x_e = x − b / √(M(x_e)² − 1)`, and the crack arrives at
`t_exit + t_flight(x_e) + b·M / (c·√(M² − 1))`. No solution means no crack:
the listener is behind the muzzle, outside the cone, or beyond where the bullet
went subsonic.

This matters for a 9 mm pistol: 115 gr from the CZ's 117 mm barrel leaves at
~350 m/s, Mach 1.02 at 20 °C. The cone is ~20° wide, the bullet goes subsonic
within ~10 m, and on a cold day (`air.temperature` < 10 °C) the same round
has a crack where a warm day has none. 147 gr subsonic loads have none at all.

Near M = 1 Whitham's slender-body result overstates the crack: the bow shock
stands off and steepens slowly. The model fades the amplitude linearly below
`M² − 1 = 0.25` (an estimate, `nwave.transonic_m2`).

## Mechanism

Each part is a bank of exponentially decaying modes `(f, τ, g)`. An impact of
speed `v` scales the bank by `v`, and its contact time
`τ_c ∝ v^{−1/5}` (Hertz) shapes which modes ring through the spectrum of a
half-sine force pulse; a broadband click of one contact length carries the
attack. Every event names the parts it rings and their weights
(`sources.EVENTS`).

Decay times are 5–20 ms. The parts press against each other and against a
hand, so a struck slide is a clack; free-bar decays of 30–60 ms made the
mechanism sound like two pipes struck together.

| Part | First modes (Hz) | Basis |
|---|---|---|
| slide | 1.15k, 2.15k, 3.4k, 5.9k, 7.9k, 11k | free-free hollow steel beam, 206 mm; torsion |
| frame | 0.62k, 1.35k, 2.6k, 4.1k, 6.8k | steel frame and grip, damped by the hand |
| barrel | 5.2k, 9.8k, 14.3k | steel tube 117 mm, OD 13, ID 9: `f₁ = 22.37/(2πL²)·√(I/A)·√(E/ρ)` |
| hammer | 4.4k, 8.8k, 13.2k | small steel part |
| trigger group | 2.9k, 6.1k, 11.8k | |
| magazine | 240 (spring), 1.8k, 3.3k, 5.2k | sheet steel box |

All of these are estimates from beam and shell formulas, not measurements.
The CZ 75 is all steel and its slide rides inside the frame rails, so it rings
longer and lower than a polymer pistol and adds a rail-sliding noise while the
slide moves (`mech.rail_noise_db`).

Each pistol is detuned once (±3 % frequency, ±20 % decay) by `--gun-seed`, so
two CZs in the game are not identical; each shot is detuned again by
`mech.detune` (±1.2 %).

## Brass

The 9×19 case is a short brass cylinder, R ≈ 4.7 mm, wall ≈ 0.4 mm at the
mouth. The ovalling modes of a thin cylinder,

```
f_n = (1/2π) · (h/R²) · √(E / 12ρ(1−ν²)) · n(n²−1)/√(n²+1)
```

give ≈ 8.5 kHz for `n = 2` and 2.83× that for `n = 3`; the model adds an
axial mode at 1.58×. A recording of a CZ 75 outdoors measures the first mode
at 10.84–10.87 kHz across three cases (the short case is stiffer than a long
shell) and its decay at ~220 dB/s, `τ ≈ 40 ms`, `Q ≈ 1300`; the model uses
those. The higher modes lie above that recording's 16 kHz MP3 limit and are
still the formula's. Mouth hits
ring fully; base hits are duller.

Flight is ballistic from the ejection port (3.6 m/s, 40° up, 100° right of the
bore by default), landing ~1–3 m to the right after ~0.8 s from 1.4 m. Each
bounce returns `e` of the normal speed (randomised for tumbling) and loses
horizontal speed. Then the case rolls: a thinning Poisson train of small
impacts convolved with a damped ring.

| Surface | e | Case ring kept | Rolls | Surface modes |
|---|---|---|---|---|
| concrete | 0.28 | all | yes | — |
| tile | 0.32 | all | yes | 1.2k, 2.6k |
| wood | 0.22 | 55 % | yes | 180, 420, 950 |
| dirt | 0.05 | 10 % | no | thud only |
| steel plate | 0.35 | all | yes | 310, 870, 1650, 2900, long |

## Variation

Variation comes from three places, and each knob has a per-shot spread:

1. Physics fed by random inputs: ammunition velocity (±8 m/s) and charge move
   the blast, Mach number, slide speed and every mechanical time and level
   together.
2. Stochastic events: secondary flash (25 %), casing trajectory, which end of
   the case hits, random modal phases.
3. Per-gun and per-shot detuning of every mode.

Variation never comes from replaying a sample with pitch jitter, so no two
shots share a waveform and their differences stay physically consistent.

## For the engine

- A shot is a list of timed events with positions, and each event is a short
  signal. The engine needs four voices per shot: the muzzle (blast, gas,
  flash), the mechanism (one point near the slide), the crack (computed per
  listener from the trajectory, not from a point), and the brass (a point that
  moves to each landing spot).
- The crack is listener-dependent geometry. Its time and amplitude must be
  solved per listener, as above; it cannot be a positioned sample.
- Directivity of the blast is part of the source; distance, air absorption and
  reflections are not, and the tool's `--room` exists only for listening.
- The blast front needs band-limiting: the tool evaluates the analytic
  waveform at 8× the output rate and low-passes.

## Sources

- R. C. Maher and the Montana State gunshot acoustics publications, among
  them <https://www.montana.edu/rmaher/publications/maher_aac_0406.pdf> and
  <https://www.montana.edu/rmaher/audio_monitor/ieeedsp_2006_maher.pdf>. Muzzle blast 3–5 ms,
  directivity, reflections, the shock-wave arrival before the blast.
- T. B. Routh, R. C. Maher, *Recording anechoic gunshot waveforms of several
  firearms at 500 kilohertz sampling rate*, ASA 2016 (Glock 19 9 mm among
  eight firearms):
  <https://www.ojp.gov/ncjrs/virtual-library/abstracts/recording-anechoic-gunshot-waveforms-several-firearms-500-kilohertz>.
  Directivity of more than 20 dB from the bore to side or rear.
- H. Hacıhabiboğlu, *Procedural Synthesis of Gunshot Sounds Based on
  Physically Motivated Models*, in *Game Dynamics*, Springer 2017, pp. 47–69:
  <https://link.springer.com/chapter/10.1007/978-3-319-53088-8_4>. Blast and
  N-wave as separate parametric layers; the Friedlander form.
- L. Mengual, D. Moffat, J. D. Reiss, *Modal Synthesis of Weapon Sounds*, AES
  61st Conference on Audio for Games, 2016:
  <https://qmro.qmul.ac.uk/xmlui/handle/123456789/13348>. Modal and additive
  synthesis of weapon sounds, judged as real as recordings in 4 of 7 cases.
- J. W. M. DuMond et al., *A determination of the wave forms and laws of
  propagation and dissipation of ballistic shock waves*, JASA 18 (1946):
  <https://authors.library.caltech.edu/46470>. N-wave amplitude ∝ b^{−3/4},
  period ∝ b^{1/4}.
- G. B. Whitham, *The flow pattern of a supersonic projectile*, Comm. Pure
  Appl. Math. 5 (1952). The N-wave formulas above.
- *Precision and accuracy of acoustic gunshot location in an urban
  environment*, arXiv:2108.07377: the Weber spark model for muzzle blast, a
  Weber radius that depends on the angle to the barrel.
- Ylikoski's handgun impulse measurements, compiled at
  <https://www.yarchive.net/gun/sound_levels.html>: 9 mm pistol 154 dB, 0.3 ms
  near the shooter.
- Gun Tests, CZ 75 B reviews: 4.6 in (117 mm) barrel, 115 gr at
  ~1130–1160 fps (345–354 m/s) from CZ 75 variants:
  <https://www.gun-tests.com/handguns/cz-75-b-sa-9mm/>.
- Cyclic rate of the Glock 18, ~1200 rounds per minute (≈50 ms per shot), as an
  upper bound on a pistol's slide cycle.

## Against a recording

A CZ 75 shot outdoors (44.1 kHz stereo, through MP3), measured with
`python3 -m gunshot.analyze`:

- the recording chain hides the blast: nothing below ~250 Hz (−45 dB), and
  automatic gain holds the first 110 ms at one level, so the direct front is
  no louder than the reflections after it (crest factor 9.8 dB over the first
  50 ms). Neither the front's shape nor the blowdown's low end can be checked
  against it;
- what is heard is the space: a dense field from 300 Hz to 5 kHz for
  ~250 ms with the highs dying first, and an echo at 280 ms (a surface
  ~48 m away). The first 3 ms are flat from 1 to 13 kHz: the front;
- the brass lands 0.86 s after the shot and again 0.40 s later; the model's
  flight gives 0.8–0.9 s for the first landing. Its ring is the one
  measurement taken into the model (above);
- the mechanism cannot be separated from the reflections in 4–60 ms.

`--recorder 20 --room room` renders through a stand-in for that chain and
that space, so a render and the recording can be compared on equal terms.
Calibrating the blast and the mechanism needs recordings made for it:
manual gain with 20 dB of headroom, lossless, a known distance and angle in
open ground, plus dry fire, a hand-racked slide and cases dropped on concrete
recorded close and separately.

## What is not known yet

- No measured CZ 75 B mechanical spectra were found. Every mode table is an
  estimate; the first job with a real pistol is a contact-microphone recording
  of each part, struck, to replace them.
- No measured pistol slide timings were found; the cycle times come from the
  two-mass model with estimated slide mass, spring and stroke.
- Whitham's formula near M = 1 is the weakest physics in the model.
- The blast's spectrum against angle is reduced to one level and one duration
  scale; measured handgun directivity changes with frequency as well.
