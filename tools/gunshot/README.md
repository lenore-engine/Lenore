# gunshot

Renders shots of a CZ 75 B from the model in `docs/audio/gunshot.md`, to tune
that model by ear before the engine carries it. Needs Python 3.10+ and numpy.

```sh
cd tools/gunshot
python3 -m gunshot --list                          # every knob, default, spread, unit
python3 -m gunshot -n 8 --normalize -o out         # 8 shots, outdoors, as a phone records them
python3 -m gunshot -n 8 --env forest --receiver ear --normalize
python3 -m gunshot -n 8 --env dry --receiver none --stems   # the dry sources, for the engine
python3 -m gunshot --preset presets/game.json -n 8 --normalize
python3 -m gunshot --set ammo.velocity=375±5 --set air.temperature=-10 -n 4
python3 -m gunshot --sequence 0.18 -n 17 --set magazine.rounds=15 --set trigger.double_action=1
python3 -m gunshot --fixed --set listener.azimuth=3 --set listener.distance=15   # no variation
```

A shot is rendered in three stages:

1. **Sources** at the listener, in pascals: blast, crack, mechanism, brass.
2. **Environment** (`--env`): the direct sound, its ground reflection, the
   field scattered back by what stands around the shooter, a distant echo
   and the terrain rolling it on. `range` (default) is fitted to the CZ 75
   recording below; `open`, `forest`, `urban`, `room` are set by hand;
   `dry` is the sources alone.
3. **Receiver** (`--receiver`): what records or hears it. `phone` (default,
   fitted) overloads, high-passes, rides an automatic gain control and cuts
   at 16 kHz; `ear` is a gentler version for a game mix; `none` writes the
   pressure, 1.0 = `mix.full_scale_pa`.

The dry blast is a 0.1 ms spike: alone it sounds like a click, and the
mechanism and brass, 30-60 dB under it, are inaudible. What anyone knows as
a gunshot is that spike scattered by the place over the next second, through
a recorder that could not take 157 dB. Stages 2 and 3 are that, and every
level in them is physical (energy re the source at 1 m, dB SPL), so moving
the listener or changing the ammunition still moves everything together.

`--set KNOB=V` sets a value and `KNOB=V±S` its per-shot spread too; presets are
JSON with the same keys, `[value, spread]` for both. `--env` and `--receiver`
apply first, then presets, then `--set`. `--seed` picks the shot-to-shot
variation, `--gun-seed` picks the individual pistol.

Each shot writes `shot_NNN.wav` (32-bit float; through a receiver the gain
control's threshold is written at -14 dBFS RMS, so levels compare across
shots unless `--normalize`), its stems with `--stems` (`blast`, `nwave`,
`mech`, `case`, each in the space and riding the mix's gain, so they sum to
the mix but for the microphone's overload), and `shot_NNN.json` with every
drawn knob and every event's time and impact speed. The console line gives
the Mach number, whether there is a crack, the blast level and the event
times relative to the bullet leaving the muzzle. `--drive DB` saturates the
result into tanh after the receiver.

```sh
python3 -m gunshot.analyze real.wav --synth out/shot_000.wav --plot cmp.png
```

measures each shot in a recording and in a render the same way: peak,
clipping, rise, positive phase, envelope, brass ticks, decay and 1/3-octave
spectra over the first 5 ms and the whole shot. With `--synth` it also gives
the distance between them: the mean dB difference of their 1/3-octave
levels in 10 ms frames, by time window (0-3, 3-20, 20-100, 100-250, 250-700,
700-1400 ms), plus crest factor and kurtosis, which say how far the blast
stands out of its tail and how clipped the tail is.

```sh
python3 -m gunshot.fit real.wav --set listener.distance=0.5 -o presets/my_place.json
python3 -m gunshot --preset presets/my_place.json -n 8 --normalize
```

fits the `env.*` and `rec.*` knobs to a recording of one shot, minimising
that distance, and writes them as a preset: the gun as modelled, heard in
that place through that recorder. Set the recording's distance and angle
first. It takes a few minutes.

| Preset | |
|---|---|
| `game` | mechanism +8 dB, brass +26 dB, so they read under the blast |
| `downrange` | 15 m downrange, 124 gr +P: the crack before the blast |
| `bystander` | 8 m to the side |
| `subsonic` | 147 gr subsonic |
| `limp_wrist` | loose grip: slower cycle, longer ring |
