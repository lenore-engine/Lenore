# Blast waves in air: what the literature gives a sound model

Background for the muzzle blast in `gunshot.md`, and for explosions later.
What follows is what could be read of each source; several are paywalled and
were seen only as abstracts, and that is said where it matters.

## The shape: Friedlander and its variants

The pressure of an ideal blast at a fixed point is a jump followed by an
exponential decay through zero into a shallower, longer suction:

```
p(t) = P · (1 − t/T₊) · exp(−b·t/T₊)
```

Friedlander (1946). The "modified" form fits the positive and negative phases
with separate decay constants, since one `b` cannot match both. Engineering
work often drops the negative phase; a sound model cannot, because the
negative phase carries most of the low-frequency energy.

The Weber model (1939, spark discharges) gives a similar waveform from one
parameter, the Weber radius `r_w`, where the outflow of the source slows to
sonic speed. ISO 17201-2 (shooting range noise; 2006, revised 2025) builds its
muzzle blast source on it, with an angle-dependent `r_w` for the barrel
(Hirsch and Bertels 2013). Energy pushes the spectrum down as `r_w` grows; a
handgun's `r_w` is ~0.4 m. The equations themselves are in the paywalled
standard and were not read.

## Scaling: from energy to peak and duration

Hopkinson–Cranz scaling: the same explosive at distances scaled by the cube
root of energy gives the same waveform, `Z = r / W^{1/3}` (W in kg TNT,
4.6 MJ/kg). Kingery–Bulmash (1984) tabulates `P`, `T₊` and impulse against
`Z`; Kinney and Graham (*Explosive Shocks in Air*, 1985) give closed forms:

```
P/P₀    = 808 (1 + (Z/4.5)²) / √((1 + (Z/0.048)²)(1 + (Z/0.32)²)(1 + (Z/1.35)²))
T₊/W^⅓  = 980 (1 + (Z/0.54)¹⁰) / ((1 + (Z/0.02)³)(1 + (Z/0.74)⁶) √(1 + (Z/6.9)²))   [ms]
```

The constants are as usually quoted, from memory and secondary sources; check
them against the book before relying on them.

A check against the pistol, with Henriksen and Cummings' finding that only
about one sixth of the energy left after the projectile's kinetic energy is
prompt enough to drive the blast:

- 0.35 g of powder at ~4 MJ/kg ≈ 1.4 kJ; the bullet takes 0.46 kJ; one
  sixth of the rest ≈ 160 J ≈ 0.035 g TNT, `W^⅓ ≈ 0.033`.
- At 1 m, `Z ≈ 31`: Kinney–Graham gives `P ≈ 2.8 kPa`. The model's on-axis
  value at 1 m is 2.5 kPa (1.1 kPa at 90° plus 7 dB). That agrees.
- The same formula gives `T₊ ≈ 0.13 ms`, against 0.3 ms measured near a 9 mm
  pistol (Ylikoski). A spectral fit to a CZ 75 recording prefers
  0.07–0.2 ms; the model now uses 0.15 ms, after 0.45 ms proved too dull.

Brode (1955) gives the same peak scaling for `Z = 0.2–2`; Reed's equations
and ANSI S2.20 extend blast estimation to long range and weather, which
belongs to propagation.

Both families come from spherical charges of grams to tonnes. At small `Z`
Kingery–Bulmash disagrees with simulation and the Friedlander form fails,
since the detonation products are still expanding.

## The most direct model: Salomons 2024

E. M. Salomons (TNO), *Analytical model for sound of explosives and
firearms*, JASA 156(3), 2034–2044, 2024 (doi:10.1121/10.0030301). An
analytical Friedlander waveform and a closed-form 1/3-octave spectrum of the
sound exposure level, from numerical spherical blast solutions spanning
10 kPa to 100 MPa, with the nonlinear lengthening of the pulse and the
resulting downward shift of the spectrum. For firearms it adds one empirical
directivity correction; otherwise everything follows from the energy.
Validated from 0.4 g to 4 kg TNT. Only the abstract was read. Its formulas
should replace the model's hand-set peak, duration and directivity scaling.

## Nonlinear propagation

A blast is a weak shock for a long way. The front steepens, the pulse
lengthens and the peak decays faster than `1/r`:

- Landau–Whitham weak-shock theory: for a spherical wave the positive phase
  grows as `√(ln(r/a))` far out, very slowly.
- Measured weak spherical N-waves under 100 Pa: `P ∝ r^−1.38`,
  `T₊ ∝ r^0.19`, rise time growing linearly with distance.
- Billot, Marinus, Harri, Moiny (Royal Military Academy, Brussels), *Evolution
  of acoustic nonlinearity in outdoor blast propagation from firearms*, JASA
  155(2), 1021–1035, 2024: eight calibres measured, a 9 mm Browning among them
  propagated to 300 m with a nonlinear solver; propagation does not become
  truly linear even there. ISO 9613 and ISO 17201 ignore this.
- The rise time of the front is set by molecular relaxation of O₂ and N₂ with
  water vapour, not by viscosity (augmented Burgers equation;
  Pestorius–Blackstock stepping). Near a gun the rise is a few microseconds;
  6 µs was measured at 3 m from a Colt .45 at 165 dB.

For the engine this means a blast source cannot be a fixed sample attenuated
by `1/r`: its length and spectrum depend on the distance it has travelled.
Two ways to carry it:

1. Analytical: emit at the listener a Friedlander whose `P` and `T₊` follow
   the Salomons or weak-shock laws for the path length (cheap, per voice).
2. Numerical: march the waveform along the path with a Burgers step
   (Pestorius–Blackstock): nonlinear distortion in the time domain, then
   absorption and dispersion in the frequency domain. Exact, costlier; for
   distant explosions.

## The parts a sound model adds to the blast

Engineering blast work stops at the overpressure. What is heard also has:

- **the expanding products.** For a gun, the barrel's blowdown, already in
  the model as a monopole `ρ/(4πr)·dQ/dt`. For a charge, the fireball's
  growth and collapse does the same job and sets the low-frequency body.
- **turbulence and combustion.** Chadwick and James, *Animating Fire with
  Sound* (SIGGRAPH 2011), drive a combustion sound model from a coarse flame
  simulation, then add the high band by bandwidth extension and texture
  synthesis. That is the route for the roar after an explosion, and for the
  secondary flash at a muzzle.
- **ground.** A surface burst sees its reflection merge with the incident
  wave (×1.8 on the peak for hemispherical bursts in the blast literature).
  For a pistol 1.4 m up the reflection arrives ~1–8 ms later and is the
  most audible part of its "boom".
- **the receiver.** At 155–165 dB with rises of microseconds, microphones
  and recorders overload, and almost every gunshot recording people know is
  clipped and compressed. A faithful model sounds thinner than a sound
  library; a game mix may want that overload back as a listener effect,
  never inside the source.

## Sources

- F. G. Friedlander, *The diffraction of sound pulses*, Proc. R. Soc. A, 1946.
- ISO 17201-2, *Acoustics — Noise from shooting ranges — Part 2*:
  <https://www.iso.org/standard/81320.html>.
- Weber model, Weber radius, angle-dependent radius: *Precision and accuracy
  of acoustic gunshot location in an urban environment*,
  <https://arxiv.org/pdf/2108.07377>.
- E. M. Salomons, JASA 156(3), 2024: <https://pubmed.ncbi.nlm.nih.gov/39324738/>.
- Billot et al., JASA 155(2), 2024:
  <https://pubs.aip.org/asa/jasa/article/155/2/1021/3261965/Evolution-of-acoustic-nonlinearity-in-outdoor>.
- Karlos, Solomos, Larcher, *Analysis of the blast wave decay coefficient
  using the Kingery–Bulmash data*, 2016:
  <https://journals.sagepub.com/doi/full/10.1177/2041419616659572>.
- *An Abridged Review of Blast Wave Parameters*, Defence Science Journal
  (Brode; modified Friedlander):
  <https://publicationsdrdo.in/index.php/dsj/article/view/1149>.
- Henriksen and Cummings, gun blast energy partition, summarised in
  <https://apps.dtic.mil/sti/pdfs/ADA051065.pdf>.
- *Shock Wave in the Beirut Explosion: Theory and Video Analysis* (√ln r law):
  <https://arxiv.org/html/2510.24742v1>.
- Molecular relaxation and sonic boom rise time (UEA):
  <https://ueaeprints.uea.ac.uk/id/eprint/71669/>.
- Reed and Church, *Sedan Long Range Blast Propagation*, Sandia 1963:
  <https://apps.dtic.mil/sti/tr/pdf/ADA438188.pdf>.
- J. N. Chadwick, D. L. James, *Animating Fire with Sound*, SIGGRAPH 2011:
  <https://research.cs.cornell.edu/Sound/fire/FireSound2011.pdf>.
- Wunderli, Pieren, Heutschi, *The Swiss shooting sound calculation model
  sonARMS*, Noise Control Eng. J. 2012: <https://empa.ch/web/s509/sonarms>.
- Recording near firearms (rise time, overload): Maher's AES papers at
  <https://www.montana.edu/rmaher/publications/>, and
  <https://www.grasacoustics.com/files/MiscFiles/Cases/CA_Noise_from_firearms.pdf>.
