# Lenore

A Vulkan 1.3 engine in Zig 0.17, built for integrated GPUs first: RADV on an
RDNA3 iGPU is the reference target. Linux on Wayland is the one platform for
now.

This repository is the engine and the umbrella over its modules. Each module is
a repository of its own and builds and tests from its own directory.
`lenore-resources` is the vocabulary the others share; beyond it, the one
module that names a sibling is `lenore-gpu`, which makes the Vulkan surface
from `lenore-platform`'s native handles.

| Module | Owns |
|---|---|
| `lenore-gpu` | everything that talks to Vulkan: device, swapchain, GPU resources, descriptor layouts, the renderer and its passes |
| `lenore-scene` | CPU-side scene state and the per-frame math over it: camera, transforms, culling, picking, shadow fit, light descriptions, exposure |
| `lenore-resources` | the neutral vocabulary two modules share without naming each other: interchange vertex, bounds, material description, sampler configuration, skeletons, clips, the 2D draw list |
| `lenore-platform` | window, input, clock and system font lookup, on native Wayland; no graphics API |
| `lenore-imui` | immediate-mode UI: regions, event routing, drawing into the `lenore-resources` draw list |
| `lenore-gltf` | glTF 2.0 parsing into `lenore-resources` values; no device |
| `lenore-text` | text shaping and glyph rasterisation over FreeType and HarfBuzz |
| `lenore-ktx` | BC7 compression and the KTX2 container; bytes in, bytes out |

The engine (`src/`) owns composition and policy: the frame loop, lifetimes, the
world and levels, fonts and the overlay host, and time policy over the
platform's clock.

## Building

Needs Zig 0.17.0 and `slangc` on `PATH`; the shaders in `assets/shaders/` are
compiled by the build. A program links `libwayland-client` and `libxkbcommon`,
and opens `libvulkan.so.1` and `libfontconfig.so.1` when it starts. It needs a
Vulkan 1.3 driver; without fontconfig it runs with no system font and no
interface text.

```sh
zig build examples                       # every example
zig build run-minimal -- model.glb       # one of them
zig build test                           # the engine's suite and the editor's
zig build run-editor
cd lenore-gpu && zig build test          # a module, from its own directory
```

The shipping configuration is `--release=fast`. To build without network,
lay the packages out in a directory and pass `--system <dir>`. On a host where
Zig cannot find the C runtime start files (a musl distribution without GCC's
`crtbegin*.o`), point `ZIG_LIBC` at a libc file.

## An application

`examples/minimal/main.zig` is the whole shape in under a hundred lines:

1. `engine.init(allocator, options)` opens the window and the device.
2. A model is loaded with `lenore-gltf`, a `Level` built from it with
   `Level.init(allocator, engine.deps(), &model, .{ .capacity = ... })`, and
   `engine.install(&level, &model)` uploads it and frames the camera.
3. Components are plain values. Each declares the hooks it wants:

   ```zig
   const Turntable = struct {
       speed: f32 = 0.5,
       pub const hooks: lenore.Hooks(Turntable) = .{ .update = update };
       fn update(self: *Turntable, engine: *lenore.Engine, _: *lenore.Level, time: lenore.FrameTime) !void {
           engine.camera.yaw += self.speed * time.delta;
       }
   };
   ```

   A misspelt hook, a wrong signature or a missing `hooks` is a compile error
   at the component.
4. `try engine.run(&level, .{ &orbit, &turntable })` runs frames until the window
   closes or a component calls `engine.requestExit()`.

### Hooks

At each point the components are called in tuple order.

| Hook | When |
|---|---|
| `ui_regions` | before the frame's events are routed; registers every interactive area, draws nothing |
| `event` | each input event, with whether the UI took it this frame |
| `resize` | after the swapchain and render targets follow the surface |
| `update` | the frame's simulation; the look, lights and camera are read after it |
| `compute` | before any rendering opens |
| `record` | inside the main pass, after the scene |
| `depth` | after the main pass, with its depth readable by compute |
| `ui_draw` | builds the overlay drawn over the finished picture |

### Lifetimes

- `run` returns with the device drained, whether it returned normally or with
  an error. Device resources an application owns may be destroyed as soon as
  it returns.
- Declare in creation order and let `defer` unwind: the engine, then the level
  (`defer level.deinit(&engine.textures)`), then the application's own device
  resources, so each is destroyed before what it was made from.
- A level is never replaced inside a hook. Call `engine.requestExit()`, then
  `engine.unload(&level)`, build and install the next one, and `run` again.
- Set `engine.look.lights` before `install`, which fits the sun's shadow to
  that block. The look is read once per frame, after `update`.

### Options worth knowing

`frame_capacity` and `material_capacity` size what a level may hold.
`interface_font = .{ .points = 14 }` opens the host's sans-serif into
`engine.interface_font`, which is null when the host has none. `render_scale`
rasterizes the scene below the window's size. `scene_shading`,
`background_shader` and `post_shader` replace the engine's own shading.
`gpu_timing` and `metering` record per-pass timings and the frame's light.

The cameras are components (`lenore.OrbitCamera`, `lenore.FlyCamera`) built on
blocks a custom controller can use directly (`lenore.MouseLook`,
`lenore.MoveKeys`).

## Examples

| Example | What it shows |
|---|---|
| `minimal` | the smallest application: a model, a light, the orbit camera, a component |
| `platform_window` | `lenore-platform` alone: a window and its events, no device |
| `gpu_core` | `lenore-gpu` alone: device memory and the staging pool under a load larger than the pool |
| `ui_panel` | the overlay on the device, with no scene |
| `ui_scene` | a widget panel over a model |
| `field` | a generated plain walked at eye height, with the frame's cost on screen |
| `blackhole` | a fullscreen effect recorded inside the main pass |
| `corpus_walker` | a viewer over a directory of glTF assets |
| `validation_app` | each rendering stage's host-side answer printed beside what the device was told, for a model |

Run any of them with `zig build run-<name>`; those that take a file say so
when started without one.

## Licence

BSD-3-Clause; see `LICENSE`, `NOTICE.md` and `CONTRIBUTING.md`.
