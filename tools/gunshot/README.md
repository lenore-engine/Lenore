# gunshot

Renders shots of a CZ 75 B from the model in `docs/audio/gunshot.md`, to tune
that model by ear before the engine carries it. Needs Python 3.10+ and numpy.

```sh
cd tools/gunshot
python3 -m gunshot --list                          # every knob, default, spread, unit
python3 -m gunshot -n 8 --stems -o out             # 8 shots, each layer separately
python3 -m gunshot --preset presets/game.json --room ground -n 8 --normalize
python3 -m gunshot --set ammo.velocity=375±5 --set air.temperature=-10 -n 4
python3 -m gunshot --sequence 0.18 -n 17 --set magazine.rounds=15 --set trigger.double_action=1
python3 -m gunshot --fixed --set listener.azimuth=3 --set listener.distance=15   # no variation
```

`--set KNOB=V` sets a value and `KNOB=V±S` its per-shot spread too; presets are
JSON with the same keys, `[value, spread]` for both. `--seed` picks the
shot-to-shot variation, `--gun-seed` picks the individual pistol.

Each shot writes `shot_NNN.wav` (32-bit float, 1.0 = `mix.full_scale_pa`, so
levels compare across shots unless `--normalize`), its stems with `--stems`
(`blast`, `nwave`, `mech`, `case`), and `shot_NNN.json` with every drawn knob
and every event's time and impact speed. The console line gives the Mach
number, whether there is a crack, the blast level and the event times relative
to the bullet leaving the muzzle.

`--room` is for listening only and is not part of the source model:
`ground` adds the ground reflection, `room` a small reverberant space.
`--drive DB` saturates the result as an ear or a recorder overloaded by a
155 dB peak does; most recorded gunshots carry that overload.

| Preset | |
|---|---|
| `game` | mechanism +8 dB, brass +26 dB, so they read under the blast |
| `downrange` | 15 m downrange, 124 gr +P: the crack before the blast |
| `bystander` | 8 m to the side |
| `subsonic` | 147 gr subsonic |
| `limp_wrist` | loose grip: slower cycle, longer ring |
