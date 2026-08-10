const std = @import("std");
const gltf = @import("lenore-gltf");
const gpu = @import("lenore-gpu");
const lenore = @import("lenore");
const platform = @import("lenore-platform");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");
const zm = @import("zmath");

const Allocator = std.mem.Allocator;
const log = std.log.scoped(.validation);

// The checks that say which stage is wrong when the picture is not what it
// should be.
//
// A wrong matrix, a wrong descriptor and a wrong barrier all produce the same
// black window. So every stage that can be answered on the host is answered
// here and printed beside what the device was told, and a run that draws
// nothing still says where it stopped agreeing.
//
// Nothing here composes a frame. The engine owns the loop, the world and the
// device; this drives it and reads what came out. What that leaves is the
// harness the file is named for.
//
// Usage: run-validation_app -- <model.glb> [environment-directory]
//
// The environment directory holds a prefiltered set in the shape the Khronos
// tool produces: `lambertian/diffuse.ktx2`, `ggx/specular.ktx2` and
// `lut_ggx.png`. Without one the scene is lit by its punctual lights alone,
// which is a supported state and not a degraded one: the black cubemaps the
// cache falls back to make every image-based term exactly zero.

const checking = std.debug.runtime_safety;
var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

// Sun shadows are on here, unlike in the corpus viewer this stage closes
// against, which switches them off so that geometric self-shadow does not
// contaminate material and normal conformance. Here the shadow is under test.
const sun_shadow_settings: scene.SunShadowSettings = .{
    .enabled = true,
    .strength = 1,
    .normal_offset_texels = 1,
};

// How many recorded frames the in-loop probes wait for. Enough that a policy
// which bakes every frame is unmistakable against one that baked once, and
// small enough that a run under `timeout` reaches it.
const probe_frames: u64 = 30;

// What one frame's rings hold. Named once, because the level is built against
// it and the check below is taken against it.
const frame_capacity: gpu.FrameCapacity = .{ .instances = 64, .joints = 256 };

// The one diagnostic sun every model is lit by. The asset's own lights are used
// only when asked for, because a viewer that lights each model the way its
// author did cannot compare two of them.
//
// Fixed rather than derived from the camera, so the same asset shades the same
// way on every run and a change to the picture is a change to the code.
//
// Two constraints decide the direction and they pull against each other. It
// travels broadly the way the camera looks, which puts the lit side of a model
// toward the viewer. But a sun travelling with the camera hides every shadow
// behind the thing casting it, so it is swung well off that axis: about 62
// degrees from the initial framing and 49 above the horizon.
// `max_sun_view_alignment` is the check that holds the two apart, so editing
// either constant reports the collapse rather than producing a picture with no
// shadows in it.
//
// Intensity is pi because the BRDF's diffuse lobe divides by pi. A white
// dielectric facing this light head-on returns exactly 0.97: 0.96 of diffuse,
// which is what the Fresnel term leaves of a white base at normal incidence,
// and 0.01 of specular. So a fallback texture still comes out very near white
// and the shading is visible in how it falls off rather than in its peak.
const key_light_direction = res.Vec3{ 0.6, -0.75, 0.28 };
const key_light_intensity = std.math.pi;

// How far the sun has to travel from the view axis for its shadows to be
// visible rather than hidden behind their casters, as the cosine between the
// two unit directions. Below this the picture is lit and shadowless, which
// reads as a broken shadow pass and is not one.
//
// The bound is a choice and not a derivation: loose enough to leave the framing
// free, tight enough that a sun straight down the barrel of the camera cannot
// pass.
const max_sun_view_alignment: f32 = 0.7;

// Failures a check found, as distinct from errors the code returned. A check
// that fails does not stop the run: the point is to reach the frame and see
// what it does, with every disagreement already named.
var failures: u32 = 0;

fn check(passed: bool, comptime what: []const u8, args: anytype) void {
    if (passed) {
        log.info("ok    " ++ what, args);
    } else {
        failures += 1;
        log.err("FAIL  " ++ what, args);
    }
}

const usage =
    "usage: validation_app [--asset-lights] [--flat-background] [--no-bloom] " ++
    "[--bloom-scatter=F] [--bloom-intensity=F] <model.glb> [environment/]";

// The value of a `--name=value` argument, or null if this is not that argument.
fn parseOption(argument: []const u8, comptime prefix: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, argument, prefix)) return null;
    return argument[prefix.len..];
}

const Arguments = struct {
    model_path: []const u8,
    environment_path: ?[]const u8 = null,

    // The asset's own lights are opted into, not out of. A viewer that lights
    // each model the way its author did cannot compare two of them: the
    // material under test changes appearance with the lighting. The flag exists
    // because an asset that ships lights is still worth seeing as authored.
    asset_lights: bool = false,
    // The background follows the environment by default: with one loaded it is
    // what the surfaces reflect, and drawing it is what makes the reflections
    // legible as reflections of something. This turns it off without giving up
    // the environment, which is what a flat field behind a conformance asset
    // asks for.
    flat_background: bool = false,
    // Bloom is on by default, because the bar this stage closes against
    // composites it in every run its viewer makes. The switch is what makes the
    // A/B one build rather than two.
    bloom: bool = true,
    bloom_settings: gpu.BloomSettings = .{},

    // The iterator is the caller's: on Windows it owns the buffer the arguments
    // were decoded into, and the paths returned here point into it.
    fn parse(iterator: *std.process.Args.Iterator) !Arguments {
        _ = iterator.skip();

        var positional: [2]?[]const u8 = .{ null, null };
        var count: usize = 0;
        var parsed: Arguments = .{ .model_path = "" };

        while (iterator.next()) |argument| {
            if (std.mem.eql(u8, argument, "--asset-lights")) {
                parsed.asset_lights = true;
                continue;
            }
            if (std.mem.eql(u8, argument, "--flat-background")) {
                parsed.flat_background = true;
                continue;
            }
            if (std.mem.eql(u8, argument, "--no-bloom")) {
                parsed.bloom = false;
                continue;
            }
            if (parseOption(argument, "--bloom-scatter=")) |value| {
                parsed.bloom_settings.scatter = std.fmt.parseFloat(f32, value) catch {
                    log.err("--bloom-scatter takes a number in [0, 1), got '{s}'", .{value});
                    return error.InvalidArgument;
                };
                continue;
            }
            if (parseOption(argument, "--bloom-intensity=")) |value| {
                parsed.bloom_settings.intensity = std.fmt.parseFloat(f32, value) catch {
                    log.err("--bloom-intensity takes a non-negative number, got '{s}'", .{value});
                    return error.InvalidArgument;
                };
                continue;
            }
            if (count == positional.len) {
                log.err(usage, .{});
                return error.TooManyArguments;
            }
            positional[count] = argument;
            count += 1;
        }

        parsed.model_path = positional[0] orelse {
            log.err(usage, .{});
            return error.MissingModelPath;
        };
        parsed.environment_path = positional[1];
        return parsed;
    }
};

// Where the camera looks. `framed` is the pose the model is fitted into and the
// one every load-time check is computed against; the six others look along a
// cube face's axis without moving the eye.
//
// They exist to make the background comparable to something. A cube face covers
// exactly ninety degrees, so a view down one axis at that field of view is the
// same picture as the face itself, and the environment can be checked against
// its own texels rather than against an impression of them. It is also what
// makes an orientation error visible at all: looking up must give what the +Y
// face holds, and a mirrored or swapped cube says so immediately.
const View = enum {
    framed,
    plus_x,
    minus_x,
    plus_y,
    minus_y,
    plus_z,
    minus_z,

    // KTX v2 specification, 3.6 faceCount: the faces are stored in this order,
    // so the digit that selects a view is also the index of the face it should
    // be held against.
    fn direction(self: View) ?res.Vec3 {
        return switch (self) {
            .framed => null,
            .plus_x => .{ 1, 0, 0 },
            .minus_x => .{ -1, 0, 0 },
            .plus_y => .{ 0, 1, 0 },
            .minus_y => .{ 0, -1, 0 },
            .plus_z => .{ 0, 0, 1 },
            .minus_z => .{ 0, 0, -1 },
        };
    }

    fn forKey(key: platform.PhysicalKey) ?View {
        return switch (key) {
            .digit_0 => .framed,
            .digit_1 => .plus_x,
            .digit_2 => .minus_x,
            .digit_3 => .plus_y,
            .digit_4 => .minus_y,
            .digit_5 => .plus_z,
            .digit_6 => .minus_z,
            else => null,
        };
    }
};

// The pose for a view, rebuilt from the framing rather than adjusted from
// wherever the camera stands. A resize reframes, so this is what a face view has
// to be reapplied through or the next one silently returns to the framed pose.
fn applyView(engine: *lenore.Engine, view: View, sphere: res.Sphere) void {
    _ = engine.frameCamera(sphere);

    const direction = view.direction() orelse return;

    // The pivot is dropped first. An orbit anchor derives the eye from the
    // angles, so aiming one moves the camera around the model instead of
    // turning it where it stands, and a view of the environment is exactly the
    // case where the eye must not move.
    engine.camera.detach();

    // Straight up and straight down leave the azimuth undefined, and `lookAt`
    // keeps whatever it was on purpose. Pinning it first makes those two views
    // reproducible between runs and independent of which view was selected
    // before.
    if (direction[0] == 0 and direction[2] == 0) engine.camera.yaw = -std.math.pi / 2.0;

    const eye = engine.camera.placement().position;
    engine.camera.lookAt(eye + direction);

    // A cube face spans ninety degrees. The window is not square, so only the
    // vertical extent matches the face exactly; the horizontal shows more or
    // less of the neighbouring faces with the aspect ratio.
    switch (engine.camera.projection) {
        .perspective => |*perspective| perspective.fov_y = std.math.pi / 2.0,
    }
}

// What drives the loop: key handling, reframing, and every probe that can only
// be answered from inside a running frame.
const Harness = struct {
    model: *const gltf.importer.Model,
    sphere: res.Sphere,
    bounds: res.Aabb,
    background: gpu.Background,
    bloom_on: bool,
    // Read once at load rather than per frame: the world cannot gain or lose an
    // animator while the loop runs, and the bake policy has to be judged
    // against the same answer the engine gave the recorder.
    casters_move: bool,

    current_view: View = .framed,
    // Two fields rather than one: the request arrives in the event drain and is
    // applied where the swapchain extent is known.
    requested_view: View = .framed,
    // Set by `n`, applied where the clip is switched. A request rather than the
    // switch itself, because the world is advanced once per frame and changing
    // what it plays inside the event drain would sample two clips into one.
    cycle_clip: bool = false,

    // The last count reported, so a rise is named at the frame it appeared on.
    // A validation error on the first frame repeats every frame afterwards, and
    // knowing it began at frame one is most of the diagnosis.
    reported_validation: u32 = 0,
    first_frame_reported: bool = false,

    shadow_reported: bool = false,
    background_reported: bool = false,
    bloom_reported: bool = false,
    fps_reported: bool = false,

    // Everything else about rigid animation is answered before the first frame,
    // and none of it can see the frame loop. A loop that never advances the
    // world, and one that builds the draw plan once instead of per frame, both
    // leave every load-time check passing and put a still picture on screen.
    // This is the only thing that separates them.
    //
    // Deliberately a quarter of the clip rather than a frame count: how many
    // frames that takes depends on the display, and a clip several seconds long
    // has barely started after sixty of them.
    motion_probe: ?usize,
    motion_span: f32,
    motion_start: ?struct { elapsed: u64, matrix: zm.Mat } = null,
    motion_reported: bool = false,

    // The same question for the weights, and it needs its own probe for the
    // same reason. The weights and not the vertices: what the prepass produced
    // is on the device and nothing here reads it back. This separates a blend
    // that is not being driven from one whose result never arrives, and only
    // the first of those is a host defect.
    weight_probe: ?usize,
    weight_span: f32,
    weight_start: ?struct { elapsed: u64, weights: [8]f32, count: usize } = null,
    weight_reported: bool = false,

    pub fn onEvent(self: *Harness, engine: *lenore.Engine, event: platform.Event) !void {
        _ = engine;
        switch (event.payload) {
            // Presses only. A repeat would reapply a view already applied, and
            // a release would undo one the moment the key came up.
            .key => |key| if (key.action == .press) {
                if (View.forKey(key.physical)) |selected| self.requested_view = selected;
                if (key.physical == .n) self.cycle_clip = true;
            },
            else => {},
        }
    }

    pub fn onResize(self: *Harness, engine: *lenore.Engine, extent: platform.Extent2D) !void {
        applyView(engine, self.current_view, self.sphere);

        const target = engine.renderer.targetExtent();
        log.info("resized: swapchain {d}x{d}, main pass target {d}x{d}", .{
            extent.width, extent.height, target.width, target.height,
        });
        check(
            target.width == extent.width and target.height == extent.height,
            "the main pass target follows the swapchain",
            .{},
        );

        const ratio = engine.aspect();
        const view_projection = try engine.camera.viewProjection(ratio);
        reportNdcExtent(view_projection, self.bounds, ratio, extent);
        reportUpAxis(gpu.vulkanClip(view_projection), self.bounds, extent);
    }
    pub fn onCompute(_: *Harness, _: *lenore.Engine, _: *lenore.Level, _: gpu.vk.CommandBuffer) !void {}
    pub fn onRecord(_: *Harness, _: *lenore.Engine, _: *lenore.Level, _: gpu.vk.CommandBuffer) !void {}

    pub fn onFrame(
        self: *Harness,
        engine: *lenore.Engine,
        level: *lenore.Level,
        time: lenore.FrameTime,
    ) !void {
        try self.applyRequests(engine, level);
        self.probeMotion(level, time);
        self.probeWeights(level, time);
        self.probeCounters(engine);

        // The frame is recorded and not yet submitted, so this is the count the
        // recording produced rather than one a later frame added to.
        const errors = gpu.validationErrorCount();
        if (errors != self.reported_validation) {
            log.err("validation errors after frame {d}: {d}", .{ engine.recorded_frames, errors });
            self.reported_validation = errors;
        }
        if (!self.first_frame_reported) {
            log.info("first frame recorded", .{});
            self.first_frame_reported = true;
        }

        if (!self.fps_reported) {
            if (engine.last_fps) |report| {
                log.info("frames: {d:.1} per second, {d:.2} ms mean, {d:.2} ms worst", .{
                    report.fps, report.mean_ms, report.worst_ms,
                });
                self.fps_reported = true;
            }
        }
    }

    // Applied inside the frame rather than in the event drain, and logged, so a
    // captured log says which direction the window held rather than leaving it
    // to whoever was watching.
    fn applyRequests(self: *Harness, engine: *lenore.Engine, level: *lenore.Level) !void {
        if (self.requested_view != self.current_view) {
            applyView(engine, self.requested_view, self.sphere);
            self.current_view = self.requested_view;
            const facing = engine.camera.placement().front;
            log.info("view: {s}, facing [{d:.3} {d:.3} {d:.3}]", .{
                @tagName(self.current_view), facing[0], facing[1], facing[2],
            });
        }

        if (!self.cycle_clip) return;
        self.cycle_clip = false;

        const count = level.world.clipCount();
        if (count == 0) return;
        const next: u16 = @intCast((@as(usize, level.world.activeClip() orelse 0) + 1) % count);
        const started = try level.world.playClip(next);
        // The travel probe measures one clip, so switching restarts it rather
        // than comparing across two.
        self.motion_start = null;
        self.motion_reported = false;
        if (level.world.node_animator) |*animator| {
            if (animator.active_clip) |clip| self.motion_span = animator.template.clips[clip].loopSpan();
        }
        log.info("clip {d} of {d} playing, {d} animator(s) took it", .{ next, count, started });
    }

    // Does the picture actually change? Read off the plan the device was handed
    // rather than off the animator, so a break anywhere between the two shows
    // up here.
    fn probeMotion(self: *Harness, level: *lenore.Level, time: lenore.FrameTime) void {
        // A clip with no span holds one pose by definition, so there is nothing
        // to separate and this would only report a still draw as broken.
        if (self.motion_span <= 0) return;
        const probe = self.motion_probe orelse return;

        const start = self.motion_start orelse {
            self.motion_start = .{ .elapsed = time.elapsed_ns, .matrix = level.world.plan.matrices[probe] };
            return;
        };
        if (self.motion_reported) return;

        const since = lenore.seconds(time.elapsed_ns - start.elapsed);
        if (since <= self.motion_span * 0.25) return;

        const moved = matrixDrift(level.world.plan.matrices[probe], start.matrix);
        log.info("rigid animation: after {d:.3} s of frames an anchored draw moved by {d:.6}", .{
            since,
            moved,
        });
        check(moved > 1e-6, "the frame loop keeps the anchored draw moving", .{});
        self.motion_reported = true;
    }

    fn probeWeights(self: *Harness, level: *lenore.Level, time: lenore.FrameTime) void {
        if (self.weight_span <= 0) return;
        const probe = self.weight_probe orelse return;

        const live = level.world.morph_animators[probe].weights;
        const carried = @min(live.len, 8);

        const start = self.weight_start orelse {
            var snapshot: [8]f32 = @splat(0);
            @memcpy(snapshot[0..carried], live[0..carried]);
            self.weight_start = .{
                .elapsed = time.elapsed_ns,
                .weights = snapshot,
                .count = carried,
            };
            return;
        };
        if (self.weight_reported) return;

        const since = lenore.seconds(time.elapsed_ns - start.elapsed);
        if (since <= self.weight_span * 0.25) return;

        var moved: f32 = 0;
        for (live[0..start.count], start.weights[0..start.count]) |now, before|
            moved = @max(moved, @abs(now - before));
        log.info("morph: after {d:.3} s of frames a weight moved by {d:.6}", .{ since, moved });
        check(moved > 1e-6, "the frame loop keeps the blend moving", .{});
        self.weight_reported = true;
    }

    // Every other check in this app runs before the first frame, so a defect
    // whose only symptom is per-frame leaves all of them green. These read what
    // the loop actually did: the counts are the renderer's own, incremented
    // where the work is recorded, so work that was asked for and did not happen
    // shows up here and nowhere else.
    fn probeCounters(self: *Harness, engine: *lenore.Engine) void {
        const recorded = engine.recorded_frames;
        if (recorded < probe_frames) return;

        if (!self.shadow_reported) {
            const bakes = engine.renderer.shadowBakes();
            log.info("sun shadow: {d} bake(s) over {d} recorded frames", .{ bakes, recorded });
            if (self.casters_move) {
                check(
                    bakes == recorded,
                    "a scene whose casters animate re-records the map every frame",
                    .{},
                );
            } else {
                check(bakes == 1, "a scene whose casters are still bakes the map once", .{});
            }
            self.shadow_reported = true;
        }

        // Unlike the bake there is no policy to check: every recording that
        // asks for a background draws one, so the number is the frame count
        // exactly, and anything else is a draw the recorder skipped.
        if (!self.background_reported) {
            const draws = engine.renderer.backgroundDraws();
            log.info("background: {d} draw(s) over {d} recorded frames", .{ draws, recorded });
            switch (self.background) {
                .environment => check(
                    draws == recorded,
                    "a background asked for every frame is recorded every frame",
                    .{},
                ),
                // Nothing at all, and this is the half a count of zero cannot
                // tell apart on its own: the load-time check is what says the
                // run was asked for one.
                .clear => check(draws == 0, "a flat background records no draw", .{}),
            }
            self.background_reported = true;
        }

        // The chain is recorded between the two passes, where nothing the app
        // can see from outside a frame reaches, so this count is the only thing
        // that says the draws happened at all.
        if (!self.bloom_reported) {
            const chains = engine.renderer.bloomChains();
            log.info("bloom: {d} chain(s) over {d} recorded frames", .{ chains, recorded });
            if (self.bloom_on) {
                check(
                    chains == recorded,
                    "bloom asked for every frame runs the chain every frame",
                    .{},
                );
            } else {
                check(chains == 0, "a run with bloom off records no chain", .{});
            }
            self.bloom_reported = true;
        }
    }
};

pub fn main(process: std.process.Init.Minimal) !void {
    const gpa = if (checking) debug_allocator.allocator() else std.heap.smp_allocator;
    defer if (checking) {
        if (debug_allocator.deinit() == .leak) log.err("host memory leaked", .{});
    };

    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // Before the first read, so the parse phase is measured from a boundary
    // that no work has crossed.
    const clock: platform.Clock = .init(io);
    var timer: lenore.PhaseTimer = .begin(clock);

    // initAllocator because on Windows there is no other. It outlives the
    // Arguments below, whose paths point into it.
    var argument_iterator: std.process.Args.Iterator = try .initAllocator(process.args, gpa);
    defer argument_iterator.deinit();
    const arguments = try Arguments.parse(&argument_iterator);

    // The loader confines every reference a document makes to a root, so it
    // takes a directory and a name inside it rather than a path. Splitting the
    // argument here is what lets the asset live anywhere while its own images
    // still cannot escape the directory it was found in.
    const model_directory = std.fs.path.dirname(arguments.model_path) orelse ".";
    const model_name = std.fs.path.basename(arguments.model_path);
    var root = try std.Io.Dir.cwd().openDir(io, model_directory, .{});
    defer root.close(io);

    // Host side first, so a broken asset is reported before a device is
    // touched.
    var loaded = try gltf.loader.open(gpa, io, root, model_name);
    defer loaded.deinit(gpa);
    var model = try gltf.importer.build(gpa, &loaded.document, loaded.directory);
    defer model.deinit(gpa);
    // The build above names the images it did not open. This app has no
    // artifact cache to answer for them, so every file-backed one is read here
    // and the decode pass sees bytes whatever form the asset came in.
    try gltf.importer.readExternalImages(gpa, io, root, &model);
    const parse_ns = timer.split();

    try reportModel(&model);
    const verify_ns = timer.split();

    var engine: lenore.Engine = undefined;
    try engine.init(gpa, .{
        .title = "Lenore validation",
        .material_capacity = @intCast(@max(model.materials.len, 1)),
    });
    defer engine.deinit();
    log.info("device: {s}", .{engine.context.deviceName()});
    const device_ns = timer.split();

    // The lights the frame is drawn with. The asset's own if it has any and the
    // flag asked for them, and otherwise one key light, because an unlit
    // picture and a black one are the same window and the point of this app is
    // to tell them apart.
    var light_block: [gpu.max_lights]gpu.LightUniform = undefined;
    const lights = try resolveLights(&model, arguments.asset_lights, &light_block);
    engine.look = .{
        .lights = lights,
        .sun_shadow = sun_shadow_settings,
        .bloom = if (arguments.bloom) arguments.bloom_settings else null,
    };
    reportLights(&model, arguments.asset_lights, lights);

    var level = try lenore.Level.init(gpa, engine.deps(), &model, .{
        .capacity = frame_capacity,
    });
    defer level.deinit(&engine.textures);

    if (arguments.environment_path) |path| {
        try level.openEnvironment(engine.deps(), path);
    } else {
        log.info("environment: none given, image-based lighting contributes zero", .{});
    }

    // What the frames draw behind the model, decided once because it does not
    // vary between them. Without an environment the cube behind it is the black
    // fallback, so the draw would put black over a black clear: the same
    // picture at the cost of one screen-covering triangle a frame.
    engine.look.background = if (arguments.flat_background or !level.hasEnvironment())
        .clear
    else
        .environment;
    log.info("background: {s}", .{@tagName(engine.look.background)});

    try engine.install(&level, &model);

    reportLevel(&engine, &level, &model);
    reportEnvironment(&level);
    reportSun(&engine, &level);
    reportBloom(&engine, arguments);
    reportTonemap(&engine);
    reportCamera(&engine, &level);
    try reportPlan(&engine, &level, &model);
    reportTextureTransforms(&model, level.packed_materials);
    reportLitFraction(model.meshes, lights);
    reportPackedNormals(model.meshes);

    const prepare_ns = timer.split();
    log.info(
        "load: parse {d:.3} s, verify {d:.3} s, device {d:.3} s, upload {d:.3} s, world {d:.3} s, environment {d:.3} s, prepare {d:.3} s, total {d:.3} s",
        .{
            lenore.seconds(parse_ns),
            lenore.seconds(verify_ns),
            lenore.seconds(device_ns),
            lenore.seconds(level.timings.upload_ns),
            lenore.seconds(level.timings.world_ns),
            lenore.seconds(level.timings.environment_ns),
            lenore.seconds(prepare_ns),
            lenore.seconds(timer.total()),
        },
    );
    log.info("staging: {d} block(s), {d} bytes resident, {d} stall(s)", .{
        engine.staging.blockCount(),
        engine.staging.residentBytes(),
        level.timings.upload_stalls,
    });
    reportDecode(&level);

    log.info("checks before the first frame: {d} failed", .{failures});
    log.info("validation errors before the first frame: {d}", .{gpu.validationErrorCount()});

    var harness: Harness = .{
        .model = &model,
        .sphere = level.world.sphere,
        .bounds = level.world.bounds,
        .background = engine.look.background,
        .bloom_on = arguments.bloom,
        .casters_move = level.world.castersMove(),
        .motion_probe = anchoredDraw(&model),
        .motion_span = activeSpan(&level),
        .weight_probe = playingMorph(&level),
        .weight_span = morphSpan(&level),
    };
    try engine.run(&level, &harness);

    log.info("presented {d} frames", .{engine.presented_frames});
    log.info("checks failed: {d}", .{failures});
    log.info("validation errors: {d}", .{gpu.validationErrorCount()});
    if (failures > 0 or gpu.validationErrorCount() > 0) return error.ValidationRunFailed;
}

// The lights a frame is drawn with, and whose they are.
fn resolveLights(
    model: *const gltf.importer.Model,
    asset_lights: bool,
    out: *[gpu.max_lights]gpu.LightUniform,
) ![]const gpu.LightUniform {
    if (asset_lights and model.lights.len > 0) return lenore.packDocumentLights(model.lights, out);

    // Through the same constructor an asset's light takes, so the direction is
    // unit by the same code rather than by the constant being written
    // carefully.
    out[0] = lenore.packLight(try .directional(
        .{ 1, 1, 1 },
        key_light_intensity,
        key_light_direction,
    ));
    return out[0..1];
}

fn reportLights(
    model: *const gltf.importer.Model,
    asset_lights: bool,
    lights: []const gpu.LightUniform,
) void {
    if (asset_lights) {
        check(
            model.lights.len <= gpu.max_lights,
            "the model's {d} light(s) fit a block of {d}; drawing with {d}",
            .{ model.lights.len, gpu.max_lights, lights.len },
        );
    } else {
        log.info(
            "lighting: one diagnostic sun; the asset's {d} light(s) are ignored (--asset-lights uses them)",
            .{model.lights.len},
        );
    }
    for (lights) |light| {
        log.info("light: {t} colour [{d:.2} {d:.2} {d:.2}] intensity {d:.3}", .{
            light.kind, light.colour[0], light.colour[1], light.colour[2], light.intensity,
        });
    }
}

// Everything the document says, before a device exists.
fn reportModel(model: *const gltf.importer.Model) !void {
    check(model.meshes.len > 0, "model holds {d} mesh(es), {d} material(s)", .{
        model.meshes.len,
        model.materials.len,
    });
    if (model.meshes.len == 0) return error.NoGeometry;

    for (model.meshes, 0..) |*mesh, index| {
        log.info("mesh {d}: {d} vertices, {d} indices, material {d}, streams {any}", .{
            index,
            mesh.vertices.len,
            mesh.indices.len,
            mesh.material,
            mesh.streams,
        });
        check(
            mesh.indices.len == 0 or mesh.indices.len % 3 == 0,
            "mesh {d} index count {d} is whole triangles",
            .{ index, mesh.indices.len },
        );
    }

    for (model.skins) |*skin| {
        log.info("skin {d}: {d} joints over {d} slots, {d} clip(s)", .{
            skin.index,
            skin.skeleton.jointCount(),
            skin.skeleton.slotCount(),
            skin.clips.len,
        });
        for (skin.clips, 0..) |*clip, index| {
            log.info("skin clip {d}: keyed from {d:.3} to {d:.3} s over {d} slots", .{
                index,
                clip.start_time,
                clip.duration,
                clip.slot_count,
            });
        }
    }

    for (model.morph_templates, 0..) |*template, index| {
        log.info("morph template {d}: {d} target(s), {d} clip(s), defaults {any}", .{
            index,
            template.targetCount(),
            template.clips.len,
            template.defaults,
        });
    }

    if (model.node_animation) |template| {
        log.info("rigid animation: {d} slot(s), {d} clip(s)", .{
            template.slotCount(),
            template.clips.len,
        });
        for (template.clips, 0..) |*clip, index| {
            log.info("rigid clip {d}: keyed from {d:.3} to {d:.3} s over {d} slot(s), {d} channel(s)", .{
                index,
                clip.start_time,
                clip.duration,
                clip.slot_count,
                clip.channels.len,
            });
        }
    }

    // An importer that resolved a template for a mesh with no deltas, or the
    // reverse, would draw a still shape and report nothing. The two are set in
    // different walks of the document, so this is what holds them together.
    for (model.meshes, 0..) |*mesh, index| {
        check(
            (mesh.morph == null) == (mesh.morph_template == null),
            "mesh {d} carries its deltas and its weights together",
            .{index},
        );
    }
}

// What the world made of the document: bounds, joints, and whether playback
// reaches anything a draw is attached to.
fn reportLevel(
    engine: *lenore.Engine,
    level: *lenore.Level,
    model: *const gltf.importer.Model,
) void {
    const world = &level.world;

    log.info("model bounds: [{d:.3} {d:.3} {d:.3}] to [{d:.3} {d:.3} {d:.3}]", .{
        world.bounds.min[0], world.bounds.min[1], world.bounds.min[2],
        world.bounds.max[0], world.bounds.max[1], world.bounds.max[2],
    });
    check(
        world.bounds.min[0] <= world.bounds.max[0] and
            world.bounds.min[1] <= world.bounds.max[1] and
            world.bounds.min[2] <= world.bounds.max[2],
        "bounds are not inverted",
        .{},
    );

    const identity = scene.worldAabb(world.bounds, zm.identity());
    check(
        approxEqual(identity.min, world.bounds.min) and approxEqual(identity.max, world.bounds.max),
        "an identity transform leaves the bounds where they were",
        .{},
    );

    log.info("bounding sphere: centre [{d:.3} {d:.3} {d:.3}] radius {d:.4}", .{
        world.sphere.centre[0],
        world.sphere.centre[1],
        world.sphere.centre[2],
        world.sphere.radius,
    });
    check(world.sphere.radius > 0, "the bounding sphere has a radius", .{});

    check(
        level.uploaded.meshes.items.len == model.meshes.len and
            level.uploaded.texture_sets.items.len == model.materials.len,
        "upload retained {d} meshes and {d} material sets",
        .{ level.uploaded.meshes.items.len, level.uploaded.texture_sets.items.len },
    );

    var vertices: usize = 0;
    var indices: usize = 0;
    var bounds_match = true;
    for (model.meshes, level.meshes) |*source, resident| {
        vertices += resident.vertex_count;
        indices += resident.index_count;
        const expected: res.Aabb = .compute(source.vertices);
        bounds_match = bounds_match and
            approxEqual(resident.bounds.box.min, expected.min) and
            approxEqual(resident.bounds.box.max, expected.max);
    }
    var source_vertices: usize = 0;
    var source_indices: usize = 0;
    for (model.meshes) |*mesh| {
        source_vertices += mesh.vertices.len;
        source_indices += mesh.indices.len;
    }
    check(
        vertices == source_vertices and indices == source_indices,
        "uploaded {d} vertices and {d} indices",
        .{ vertices, indices },
    );
    check(bounds_match, "every uploaded mesh reproduces its source box", .{});

    // glTF 2.0 specification, 3.7.3.3: at the bind pose each joint matrix is
    // the same placement, whatever that placement is. A transpose, a wrong slot
    // or applying the inverse bind on the wrong side breaks this before all
    // three become the same torn mesh on screen.
    for (world.skins) |*skin| {
        const joints = skin.animator.jointTransforms();
        if (joints.len == 0) continue;
        var worst: f32 = 0;
        for (joints) |joint| worst = @max(worst, matrixDrift(joint, joints[0]));
        log.info("bind pose: skin {d} joints disagree by at most {d:.6}", .{ skin.source_index, worst });
        log.info("bind pose: skin {d} moves the model's height axis to [{d:.3} {d:.3} {d:.3}]", .{
            skin.source_index,
            joints[0][2][0],
            joints[0][2][1],
            joints[0][2][2],
        });
        check(worst < 1e-4, "every joint of skin {d} carries the same bind placement", .{skin.source_index});
    }

    // The weighted sum of joint matrices applied to a vertex, exactly as the
    // shader computes it. Written twice on purpose: this is the host's account
    // of what the device should produce, and the two agreeing is the check.
    for (model.meshes, world.skin_of_mesh, 0..) |*mesh, maybe_skin, index| {
        const skin_index = maybe_skin orelse continue;
        const joints = world.skins[skin_index].animator.jointTransforms();
        if (joints.len == 0) continue;

        var worst: f32 = 0;
        for (mesh.vertices) |vertex| {
            const skinned = skinnedPosition(joints, vertex);
            const placed = zm.mul(
                zm.f32x4(vertex.position[0], vertex.position[1], vertex.position[2], 1),
                joints[0],
            );
            inline for (0..3) |axis| worst = @max(worst, @abs(skinned[axis] - placed[axis]));
        }
        check(
            worst < 1e-3,
            "mesh {d} bind pose moves every vertex by that placement alone",
            .{index},
        );
    }

    check(
        world.joint_total <= frame_capacity.joints,
        "the frame's {d} joint(s) fit an array of {d}",
        .{ world.joint_total, frame_capacity.joints },
    );

    if (level.morph_overflow > 0) {
        log.warn("morph: {d} mesh(es) did not fit the prepass and draw at their bind shape", .{
            level.morph_overflow,
        });
    }
    var morphed: usize = 0;
    for (model.meshes) |*mesh| morphed += @intFromBool(mesh.morph != null);
    if (morphed > 0) {
        log.info("morph: {d} of {d} mesh(es) carry shape targets", .{ morphed, model.meshes.len });
        check(
            engine.morph_pass.registrationCount() == morphed - level.morph_overflow,
            "every morphed mesh that fit reached the prepass",
            .{},
        );
    }

    // Three states put the same still shape on screen and this separates them:
    // a document whose weights nothing animates, an animator that was never
    // started, and one that is started and never advanced. The last is the
    // frame loop's and has its own probe there; the middle one is this.
    var animated: usize = 0;
    var playing: usize = 0;
    for (model.morph_templates, world.morph_animators) |*template, *animator| {
        if (template.clips.len > 0) animated += 1;
        if (animator.active_clip != null) playing += 1;
    }
    if (animated > 0) {
        log.info("morph: {d} of {d} template(s) are playing a clip", .{ playing, animated });
        check(playing == animated, "every morph template with a clip is playing one", .{});
    }

    reportRigidAnimation(level);
    reportMaterials(level, model);
}

// Whether playback moves anything a draw is attached to. Three states put the
// same still picture on the screen and this separates them: a document with
// nothing to drive, an animator that is never advanced, and an animator that
// moves slots no geometry is anchored to.
fn reportRigidAnimation(level: *lenore.Level) void {
    const world = &level.world;
    const animator = if (world.node_animator) |*value| value else return;
    const clip_index = animator.active_clip orelse return;
    const span = animator.template.clips[clip_index].loopSpan();

    // The clip is walked and the largest departure from the bind pose is kept
    // per slot. A single sample would not do: a clip loops, so at the end of
    // its span every slot is back where it started, and comparing only there
    // reports a working clip as frozen.
    const allocator = level.allocator;
    const bind = allocator.dupe(zm.Mat, animator.world_transforms) catch return;
    defer allocator.free(bind);
    const drift = allocator.alloc(f32, animator.world_transforms.len) catch return;
    defer allocator.free(drift);
    @memset(drift, 0);
    const travel = allocator.alloc(f32, animator.world_transforms.len) catch return;
    defer allocator.free(travel);
    @memset(travel, 0);

    const steps = 16;
    for (0..steps) |_| {
        animator.update(span / @as(f32, steps));
        for (bind, animator.world_transforms, drift, travel) |before, after, *worst, *distance| {
            worst.* = @max(worst.*, matrixDrift(after, before));
            // Row three is the translation in the row vector convention this
            // project composes in.
            const offset = after[3] - before[3];
            distance.* = @max(distance.*, @sqrt(@reduce(.Add, offset * offset)));
        }
    }

    var worst_slot: f32 = 0;
    for (drift) |value| worst_slot = @max(worst_slot, value);
    log.info("rigid animation: the clip moves a slot by at most {d:.6} over {d:.3} s", .{
        worst_slot,
        span,
    });
    check(span <= 0 or worst_slot > 1e-6, "playing the clip moves at least one slot", .{});

    // An animator whose moving slots carry no geometry draws exactly the same
    // picture as no animation at all. Counted over draws rather than slots,
    // because a draw is what reaches the screen.
    //
    // No draw being anchored is legitimate rather than a fault. A skin's joints
    // are document nodes and its clip targets them, so they earn rigid slots
    // too, while the skinned mesh that follows them is never anchored: its
    // motion is in the joint matrices.
    var anchored: usize = 0;
    var moved: usize = 0;
    var farthest: f32 = 0;
    for (world.meshes) |*mesh| {
        const slot = mesh.anchor orelse continue;
        anchored += 1;
        if (drift[slot] > 1e-6) moved += 1;
        farthest = @max(farthest, travel[slot]);
    }
    log.info("rigid animation: {d} of {d} draw(s) anchored, {d} of those move", .{
        anchored,
        world.meshes.len,
        moved,
    });
    if (anchored == 0) {
        log.info(
            "rigid animation: no draw reads the {d} slot(s); the document animates joints only",
            .{animator.world_transforms.len},
        );
    }
    check(
        span <= 0 or anchored == 0 or moved > 0,
        "every frame that anchors a draw moves at least one of them",
        .{},
    );

    // The camera is framed on the bind pose, so an asset whose animation
    // carries a draw further than the sphere plays partly outside the view.
    // Not a failure: framing on swept bounds is a choice the corpus walk has to
    // make and neither reference makes it. Reported so a part leaving the
    // window is a number here rather than a surprise on screen.
    if (farthest > 0) {
        log.info("rigid animation: an anchored draw travels {d:.3} from a framed radius of {d:.3}", .{
            farthest,
            world.sphere.radius,
        });
    }

    // Back to the start, so the first frame draws the pose the checks above
    // were taken against.
    animator.play(clip_index) catch {};
}

fn reportMaterials(level: *lenore.Level, model: *const gltf.importer.Model) void {
    // A factor-only diagnostic through the host mirror of the fragment shader.
    // Unit samples leave every factor unchanged; texture variation is visible
    // in the rendered image rather than reducible to one material value here.
    const neutral: gpu.Shading.MaterialSamples = .{
        .base_colour = @splat(1),
        .metallic_roughness = @splat(1),
    };
    const first = gpu.Shading.Surface.fromMaterial(level.packed_materials[0], neutral);
    var differ = false;

    for (level.packed_materials, 0..) |record, index| {
        const surface = gpu.Shading.Surface.fromMaterial(record, neutral);
        const emitted = gpu.Shading.emissive(record, @splat(1));
        differ = differ or
            !std.meta.eql(surface, first) or
            !std.meta.eql(alphaCoverageKey(record), alphaCoverageKey(level.packed_materials[0])) or
            transformsDiffer(record, level.packed_materials[0]);
        log.info(
            "material {d}: base [{d:.3} {d:.3} {d:.3}], metallic {d:.3}, roughness {d:.3}, emissive factor [{d:.3} {d:.3} {d:.3}]",
            .{
                index,
                surface.base_colour[0],
                surface.base_colour[1],
                surface.base_colour[2],
                surface.metallic,
                surface.roughness,
                emitted[0],
                emitted[1],
                emitted[2],
            },
        );
    }

    // Whether this asset can show that the index selects anything. Materials
    // that pack to the same record are drawn identically however the indexing
    // behaves, so a picture from such a model proves nothing about it and the
    // run should say so rather than pass quietly.
    if (model.materials.len > 1) {
        check(
            differ,
            "the {d} materials differ in what a draw of them reads, so the index is observable",
            .{model.materials.len},
        );
    }
}

fn reportEnvironment(level: *lenore.Level) void {
    const loaded = if (level.environment) |*value| value else return;

    log.info("environment: lambertian {d}x{d} {d} level(s), ggx {d}x{d} {d} level(s), lut {d}x{d}", .{
        loaded.environment.lambertian.width,      loaded.environment.lambertian.height,
        loaded.environment.lambertian.mip_levels, loaded.environment.ggx.width,
        loaded.environment.ggx.height,            loaded.environment.ggx.mip_levels,
        loaded.environment.lut.width,             loaded.environment.lut.height,
    });
    for (loaded.cubes) |cube| {
        log.info("environment: {s} staged through {d} block(s) and {d} stall(s)", .{
            cube.key,
            cube.blocks,
            cube.stalls,
        });
    }
    check(
        loaded.environment.ggx.mip_levels > 1,
        "the ggx chain has {d} roughness step(s)",
        .{loaded.environment.ggx.mip_levels},
    );
    check(
        loaded.lut.varies(),
        "the lookup table varies in both channels: scale {d}..{d}, bias {d}..{d}",
        .{ loaded.lut.scale[0], loaded.lut.scale[1], loaded.lut.bias[0], loaded.lut.bias[1] },
    );
}

fn reportSun(engine: *lenore.Engine, level: *lenore.Level) void {
    const sun = if (engine.sun) |value| value else {
        log.info("sun shadow: no directional light, nothing is fitted", .{});
        return;
    };

    // The failure this catches is not a wrong picture but an uninformative one:
    // a sun aligned with the view lights every surface the camera can see and
    // puts every shadow behind its caster, so the pass looks broken and is not.
    // Measured against the initial framing, which is what a screenshot shows.
    const front = engine.camera.placement().front;
    const travel = engine.look.lights[sun.index].direction;
    const alignment = front[0] * travel[0] + front[1] * travel[1] + front[2] * travel[2];
    log.info("sun shadow: sun to view axis {d:.1} degrees", .{
        std.math.radiansToDegrees(std.math.acos(std.math.clamp(alignment, -1, 1))),
    });
    check(
        alignment < max_sun_view_alignment,
        "the sun is far enough off the view axis for its shadows to be seen",
        .{},
    );

    const size = engine.renderer.shadowMapSize();
    log.info("sun shadow: {d}x{d} map, light {d}, one texel is {d:.5} world units", .{
        size,
        size,
        sun.index,
        sun.fit.texel_world_size,
    });
    // The offset that keeps a surface from shadowing itself is authored in
    // texels and converted through the fit, so it follows scene scale without
    // being retuned. Reported because it is the number to move if the host run
    // shows acne or a shadow detached from its caster.
    log.info("sun shadow: normal offset {d:.5} world units, strength {d:.2}", .{
        sun_shadow_settings.normalOffsetWorld(&sun.fit),
        sun_shadow_settings.clampedStrength(),
    });
    check(
        sun.fit.texel_world_size > 0 and std.math.isFinite(sun.fit.texel_world_size),
        "the sun shadow fit has a texel size",
        .{},
    );

    log.info("sun shadow: casters {s}, so the map is re-recorded {s}", .{
        if (level.world.castersMove()) "animate" else "are still",
        if (level.world.castersMove()) "every frame" else "once",
    });
}

fn reportBloom(engine: *lenore.Engine, arguments: Arguments) void {
    // The chain the renderer built, against the same arithmetic stated
    // independently of it. They read one window, so a disagreement is a chain
    // built for an extent the target does not have.
    const base = gpu.bloomBaseExtent(engine.renderer.targetExtent());
    const levels = engine.renderer.bloomLevels();
    log.info("bloom: {s}, chain {d}x{d} over {d} level(s), finest {d}x{d}", .{
        if (arguments.bloom) "on" else "off",
        base.width,
        base.height,
        levels,
        gpu.mipExtent(base.width, levels - 1),
        gpu.mipExtent(base.height, levels - 1),
    });
    check(
        levels == gpu.bloomChainDepth(base),
        "the chain is as deep as this target's extent allows",
        .{},
    );

    const settings = engine.look.bloom orelse return;
    const look = gpu.bloomResolve(settings, levels) catch return;
    log.info(
        "bloom look: threshold {d:.2}, knee {d:.2}, scatter {d:.2}, intensity {d:.2}, composite {d:.4}",
        .{ look.threshold, look.knee, look.scatter, settings.intensity, look.composite },
    );

    // The load-time partner of the probe in the loop. That probe reads a count
    // of recorded chains, and a chain of one level records the whole downsample
    // half and no upsample at all, so the count would match the frame count
    // while the half this pass exists for never ran.
    check(levels >= 2, "the chain is deep enough for the upsample to have a step to run", .{});

    // A uniformly bright field passes through at the intensity asked for
    // whatever depth the window allowed. Stated here as well as in the module's
    // tests because the level count is this run's, not a chosen one.
    var summed: f32 = 0;
    for (0..levels) |level| summed += std.math.pow(f32, look.scatter, @floatFromInt(level));
    check(
        @abs(summed * look.composite - settings.intensity) <= 1e-4,
        "the composite normalizes the {d}-level chain back to the intensity",
        .{levels},
    );
}

fn reportTonemap(engine: *lenore.Engine) void {
    // The visible field carries HDR values in every channel. A copied post
    // leaves them above display range; the selected operator maps them to the
    // bounded blue field printed beside the source.
    engine.renderer.clear_colour = .{ 1.25, 2, 3, 1 };
    const mapped = gpu.toneMap(.{
        engine.renderer.clear_colour[0],
        engine.renderer.clear_colour[1],
        engine.renderer.clear_colour[2],
    }, engine.look.post) catch return;

    log.info(
        "diagnostic: HDR clear [{d:.2} {d:.2} {d:.2}] maps to [{d:.3} {d:.3} {d:.3}] at exposure {d:.2}",
        .{
            engine.renderer.clear_colour[0],
            engine.renderer.clear_colour[1],
            engine.renderer.clear_colour[2],
            mapped[0],
            mapped[1],
            mapped[2],
            engine.look.post.exposure,
        },
    );
    check(
        engine.renderer.clear_colour[0] > 1 and
            engine.renderer.clear_colour[1] > 1 and
            engine.renderer.clear_colour[2] > 1 and
            mapped[0] < mapped[1] and
            mapped[1] < mapped[2] and
            mapped[2] < 1,
        "the diagnostic separates copied HDR from bounded tonemapping",
        .{},
    );

    log.info("attachments: hdr {t}, depth {t}, present {t}", .{
        engine.renderer.hdr.format,
        engine.renderer.depth.format,
        engine.swapchain.surface_format.format,
    });
    log.info("ring strides: camera {d} B, instances {d} B, lights {d} B", .{
        engine.renderer.frame.camera.stride,
        engine.renderer.frame.instances.stride,
        engine.renderer.frame.lights.stride,
    });
    for (0..2) |index| {
        log.info("frame {d}: dynamic offsets {any}", .{ index, engine.renderer.frame.dynamicOffsets(index) });
    }
    check(
        engine.renderer.frame.dynamicOffsets(0)[0] != engine.renderer.frame.dynamicOffsets(1)[0],
        "the two frames address different slots",
        .{},
    );
}

fn reportCamera(engine: *lenore.Engine, level: *lenore.Level) void {
    const ratio = engine.aspect();
    const placement = engine.camera.placement();
    const view_projection = engine.camera.viewProjection(ratio) catch return;
    log.info("camera: eye [{d:.3} {d:.3} {d:.3}] aspect {d:.4}", .{
        placement.position[0],
        placement.position[1],
        placement.position[2],
        ratio,
    });

    // The one check a black window cannot hide behind: where a corner of the
    // model lands in clip space, computed with the same matrix the shader is
    // handed. Every visible corner has |x| and |y| within w and z in [0, w].
    var inside: u32 = 0;
    for (aabbCorners(level.world.bounds)) |corner| {
        const clip = zm.mul(zm.f32x4(corner[0], corner[1], corner[2], 1), view_projection);
        if (clip[3] > 0 and
            @abs(clip[0]) <= clip[3] and
            @abs(clip[1]) <= clip[3] and
            clip[2] >= 0 and clip[2] <= clip[3]) inside += 1;
    }
    log.info("clip space: {d} of 8 bounding corners inside the view volume", .{inside});
    check(inside == 8, "the whole model is in front of the camera", .{});

    const extent = engine.swapchain.currentExtent();
    reportNdcExtent(view_projection, level.world.bounds, ratio, extent);
    reportUpAxis(gpu.vulkanClip(view_projection), level.world.bounds, extent);

    const frustum: scene.Frustum = .fromViewProj(view_projection);
    check(frustum.intersectsAabb(level.world.bounds), "the frustum accepts the model's bounds", .{});
}

// The draw plan the first frame will record, and the host's own reading of the
// invariants the recorder refuses to draw without.
fn reportPlan(
    engine: *lenore.Engine,
    level: *lenore.Level,
    model: *const gltf.importer.Model,
) !void {
    const world = &level.world;
    // The camera this frame would be drawn with, which is also the one it is
    // culled against.
    const view_projection = try engine.camera.viewProjection(engine.aspect());
    try world.plan.rebuild(
        world.meshes,
        if (world.node_animator) |*animator| animator else null,
        world.placement,
        engine.camera.placement().position,
        view_projection,
    );

    var blended: usize = 0;
    var culled: usize = 0;
    for (world.plan.keys) |key| {
        blended += @intFromBool(key.layer == .blended);
        culled += @intFromBool(!key.visible);
    }
    log.info("draw order: {d} solid then {d} blended, back to front", .{
        world.plan.ordered.len - blended,
        blended,
    });
    log.info("culling: {d} of {d} draws outside the frustum, {d} visible batches of {d}", .{
        culled,
        world.plan.ordered.len,
        world.plan.visible_records,
        world.plan.records.len,
    });

    // The layer rule holds among the visible draws and nowhere else: the culled
    // half is recorded only by the shadow bake, which writes depth and
    // composites nothing.
    var partitioned = true;
    var farthest_first = true;
    var visible_first = true;
    var seen_blended = false;
    var seen_culled = false;
    var previous: f32 = std.math.inf(f32);
    for (world.plan.ordered) |draw| {
        const key = world.plan.keys[draw];
        if (!key.visible) {
            seen_culled = true;
            continue;
        }
        if (seen_culled) visible_first = false;
        switch (key.layer) {
            .solid => if (seen_blended) {
                partitioned = false;
            },
            .blended => {
                if (seen_blended and key.depth > previous) farthest_first = false;
                previous = key.depth;
                seen_blended = true;
            },
        }
    }
    check(visible_first, "every visible draw precedes every culled one", .{});
    check(partitioned, "every visible blended draw follows every visible solid one", .{});
    check(farthest_first, "the blended run descends in distance from the eye", .{});

    // A batch names a material, and so does every instance record it draws. The
    // two are written from different walks, one over the ordered draws and one
    // over the coalesced runs, and nothing but this compares them. When they
    // disagree the picture is plausible and wrong: each draw shades with
    // another draw's material.
    var translation_matches = true;
    var instances_match = true;
    var non_zero_first_instance = false;
    for (world.plan.batches, world.plan.records, 0..) |planned, record, index| {
        for (world.plan.instances[planned.first_instance..][0..planned.instance_count]) |instance| {
            if (instance.material_index != planned.material) instances_match = false;
        }
        const culling_matches = switch (planned.face_culling) {
            .none => !record.cull_mode.front_bit and !record.cull_mode.back_bit,
            .back => record.cull_mode.back_bit and !record.cull_mode.front_bit,
            .front => record.cull_mode.front_bit and !record.cull_mode.back_bit,
        };
        translation_matches = translation_matches and
            record.mesh == planned.mesh and
            record.material_index == planned.material and
            record.first_instance == planned.first_instance and
            record.instance_count == planned.instance_count and
            culling_matches;
        non_zero_first_instance = non_zero_first_instance or record.first_instance > 0;
        log.info("batch {d}: material {d}, instances [{d}..{d}), culling {t}", .{
            index,
            planned.material,
            planned.first_instance,
            planned.first_instance + planned.instance_count,
            planned.face_culling,
        });
    }
    check(translation_matches, "scene batches survive the explicit GPU translation", .{});
    check(instances_match, "each batch's instance records name the batch's material", .{});
    if (world.plan.records.len > 1) {
        check(
            non_zero_first_instance,
            "a multi-batch frame records a non-zero firstInstance",
            .{},
        );
    }

    // The instance records are what the vertex stage reads the index out of,
    // and in this scene every model matrix is the same. Distinct indices over
    // equal matrices are what make the picture answer which record a draw
    // selected: with one matrix the geometry cannot show it, and the material
    // can.
    if (model.materials.len > 1) {
        var distinct = false;
        for (world.plan.instances[1..]) |instance| {
            if (instance.material_index != world.plan.instances[0].material_index) distinct = true;
        }
        check(distinct, "the instance records carry more than one material index", .{});
    }

    // Whether advancing playback reaches the instance records the device reads.
    // Everything else measures the animator's own arrays; this is the only
    // thing that follows a slot through the plan and out into what is uploaded,
    // and it is what a plan built once and never rebuilt fails.
    const animator = if (world.node_animator) |*value| value else return;
    const clip_index = animator.active_clip orelse return;
    const span = animator.template.clips[clip_index].loopSpan();

    const at_rest = try level.allocator.dupe(zm.Mat, world.plan.matrices);
    defer level.allocator.free(at_rest);

    animator.update(span * 0.25);
    try world.plan.rebuild(
        world.meshes,
        animator,
        world.placement,
        engine.camera.placement().position,
        try engine.camera.viewProjection(engine.aspect()),
    );

    var anchored: usize = 0;
    var changed: usize = 0;
    for (world.meshes, at_rest, world.plan.matrices) |*mesh, before, after| {
        if (mesh.anchor == null) continue;
        anchored += 1;
        if (matrixDrift(after, before) > 1e-6) changed += 1;
    }

    // Back to the pose the frame loop starts from.
    try animator.play(clip_index);
    try world.plan.rebuild(
        world.meshes,
        animator,
        world.placement,
        engine.camera.placement().position,
        try engine.camera.viewProjection(engine.aspect()),
    );

    if (anchored > 0) {
        log.info("rigid animation: {d} of {d} anchored draw(s) change their instance matrix", .{
            changed,
            anchored,
        });
        check(span <= 0 or changed > 0, "advancing the clip rewrites an instance matrix", .{});
    }
}

fn reportDecode(level: *lenore.Level) void {
    const report = &level.image_report;
    const megapixels = @as(f64, @floatFromInt(report.pixels)) / 1e6;
    const cpu = lenore.seconds(report.decode_ns);
    const wall = lenore.seconds(report.wall_ns);

    for (report.records, 0..) |record, index| {
        if (record.decodes == 0) continue;
        log.info("image {d}: decoded {d}x{d} for {d} slot(s), alpha {d}..{d}", .{
            index,
            record.width,
            record.height,
            record.slots,
            record.alpha_low,
            record.alpha_high,
        });
    }
    log.info("decode: {d} call(s) over {d} distinct image(s), {d:.1} Mpix from {d:.1} MB", .{
        report.calls(),
        report.distinct(),
        megapixels,
        @as(f64, @floatFromInt(report.source_bytes)) / (1 << 20),
    });
    // Wall against the pool's summed CPU: their ratio is the concurrency the
    // run actually got, and the sum alone reads as a regression.
    log.info(
        "decode: {d:.3} s of CPU across the pool in {d:.3} s of wall ({d:.2}x), {d:.1} Mpix/s delivered",
        .{ cpu, wall, if (wall > 0) cpu / wall else 0, if (wall > 0) megapixels / wall else 0 },
    );
    log.info("decode: {d:.1} Mpix of that was decoded more than once", .{
        @as(f64, @floatFromInt(report.redundant_pixels)) / 1e6,
    });
    log.info("decode: window peaked at {d:.1} MB of {d:.1} MB", .{
        @as(f64, @floatFromInt(report.peak_window_bytes)) / (1 << 20),
        @as(f64, @floatFromInt(lenore.decode_window_bytes)) / (1 << 20),
    });

    // Wall, not the pool's summed CPU: the question is how much of the phase
    // the image pass occupies, and the summed figure exceeds the phase it sits
    // in as soon as more than one thread decodes.
    log.info("load: decoding is {d:.3} s of the {d:.3} s upload phase", .{
        wall,
        lenore.seconds(level.timings.upload_ns),
    });

    // The property the image-first order exists for. It is not a statement
    // about this asset: any document whose materials share an image breaks it
    // the moment a pass over materials decodes instead of a pass over images.
    check(
        report.calls() == report.distinct(),
        "each of the {d} image(s) the materials name was decoded once",
        .{report.distinct()},
    );
}

fn anchoredDraw(model: *const gltf.importer.Model) ?usize {
    for (model.meshes, 0..) |*mesh, index| {
        if (mesh.anchor != null) return index;
    }
    return null;
}

fn activeSpan(level: *lenore.Level) f32 {
    const animator = if (level.world.node_animator) |*value| value else return 0;
    const clip = animator.active_clip orelse return 0;
    return animator.template.clips[clip].loopSpan();
}

fn playingMorph(level: *lenore.Level) ?usize {
    for (level.world.morph_animators, 0..) |*animator, index| {
        if (animator.active_clip != null) return index;
    }
    return null;
}

fn morphSpan(level: *lenore.Level) f32 {
    const index = playingMorph(level) orelse return 0;
    const animator = &level.world.morph_animators[index];
    return animator.template.clips[animator.active_clip.?].loopSpan();
}

// The largest element-wise difference between two transforms. It mixes rotation
// with translation, so it is not a distance and is not comparable to anything in
// world units. It answers one question: whether two poses are the same pose.
fn matrixDrift(a: zm.Mat, b: zm.Mat) f32 {
    var worst: f32 = 0;
    inline for (0..4) |row| {
        inline for (0..4) |column| {
            worst = @max(worst, @abs(a[row][column] - b[row][column]));
        }
    }
    return worst;
}

// The weighted sum of joint matrices applied to a vertex, exactly as
// `skinnedVertexMain` computes it.
fn skinnedPosition(joints: []const zm.Mat, vertex: res.Vertex3D) [3]f32 {
    var skin: zm.Mat = .{ zm.f32x4s(0), zm.f32x4s(0), zm.f32x4s(0), zm.f32x4s(0) };
    inline for (0..4) |lane| {
        const weight = zm.f32x4s(vertex.weights[lane]);
        const joint = joints[vertex.joints[lane]];
        inline for (0..4) |row| skin[row] += joint[row] * weight;
    }

    const position = zm.f32x4(vertex.position[0], vertex.position[1], vertex.position[2], 1);
    const skinned = zm.mul(position, skin);
    return .{ skinned[0], skinned[1], skinned[2] };
}

// How much of the model faces a light, counted the way the fragment shader
// decides it: a vertex is lit when its normal has a positive dot with the
// direction toward some light.
//
// Vertex normals rather than fragments, so this is an estimate. It answers the
// only question a black window raises here, which is whether the geometry is
// pointing at the light at all.
fn reportLitFraction(meshes: []const gltf.importer.Mesh, lights: []const gpu.LightUniform) void {
    var lit: usize = 0;
    var total: usize = 0;
    for (meshes) |mesh| {
        total += mesh.vertices.len;
        for (mesh.vertices) |vertex| {
            for (lights) |light| {
                const to_light: res.Vec3 = switch (light.kind) {
                    .directional => .{ -light.direction[0], -light.direction[1], -light.direction[2] },
                    .point, .spot => .{
                        light.position[0] - vertex.position[0],
                        light.position[1] - vertex.position[1],
                        light.position[2] - vertex.position[2],
                    },
                };
                if (@reduce(.Add, vertex.normal * to_light) > 0) {
                    lit += 1;
                    break;
                }
            }
        }
    }
    log.info("lighting: {d} of {d} vertices face a light", .{ lit, total });
    check(lit > 0, "some of the model faces a light", .{});
}

// The largest distance between an authored normal and the same normal after the
// round trip through the packed vertex, over every mesh the frame uploads.
//
// Vulkan specification, Fixed-Point Data Conversion: a 10-bit snorm reads back
// as max(c / 511, -1), so a component resolves to about 1/511 and the worst a
// direction can move is a little over that.
fn reportPackedNormals(meshes: []const gltf.importer.Mesh) void {
    var worst: f32 = 0;
    var worst_mesh: usize = 0;
    var worst_at: usize = 0;
    for (meshes, 0..) |mesh, mesh_index| {
        for (mesh.vertices, 0..) |vertex, index| {
            // Through `packVertex`, which is what the upload calls. Packing the
            // components directly would test the arithmetic and not the path.
            const word = gpu.packVertex(&vertex).normal;
            const unpacked = res.Vec3{
                unpackSnorm10(word),
                unpackSnorm10(word >> 10),
                unpackSnorm10(word >> 20),
            };
            const drift = @sqrt(@reduce(.Add, (unpacked - vertex.normal) * (unpacked - vertex.normal)));
            if (drift > worst) {
                worst = drift;
                worst_mesh = mesh_index;
                worst_at = index;
            }
        }
    }

    const first = &meshes[0].vertices[0];
    const source = meshes[worst_mesh].vertices[worst_at];
    log.info("normals: first source [{d:.3} {d:.3} {d:.3}] packs to 0x{X:0>8} at byte {d} of {d}", .{
        first.normal[0],
        first.normal[1],
        first.normal[2],
        gpu.packVertex(first).normal,
        @offsetOf(gpu.GpuVertex, "normal"),
        @sizeOf(gpu.GpuVertex),
    });
    log.info("normals: worst packing drift {d:.5} at mesh {d} vertex {d}, source [{d:.3} {d:.3} {d:.3}]", .{
        worst,
        worst_mesh,
        worst_at,
        source.normal[0],
        source.normal[1],
        source.normal[2],
    });
    // Three components each within one step of 1/511, so the bound is the
    // length of that: sqrt(3)/511.
    check(worst < @sqrt(3.0) / 511.0, "the packed normals survive the round trip", .{});
}

fn unpackSnorm10(word: u32) f32 {
    const bits: u10 = @truncate(word);
    const signed: i10 = @bitCast(bits);
    return @max(@as(f32, @floatFromInt(signed)) / 511.0, -1.0);
}

fn aabbCorners(box: res.Aabb) [8]res.Vec3 {
    var out: [8]res.Vec3 = undefined;
    for (&out, 0..) |*corner, index| {
        corner.* = .{
            if (index & 1 == 0) box.min[0] else box.max[0],
            if (index & 2 == 0) box.min[1] else box.max[1],
            if (index & 4 == 0) box.min[2] else box.max[2],
        };
    }
    return out;
}

fn approxEqual(a: res.Vec3, b: res.Vec3) bool {
    return @reduce(.And, @abs(a - b) < @as(res.Vec3, @splat(1e-5)));
}

fn isIdentityTransform(uv: res.MaterialInfo.TextureMaps.UvTransform) bool {
    return uv.offset[0] == 0 and uv.offset[1] == 0 and
        uv.rotation == 0 and
        uv.scale[0] == 1 and uv.scale[1] == 1;
}

// What makes two materials draw differently through the alpha modes, which is
// the part of a material the shaded surface does not carry. glTF 2.0 section
// 3.9.4 states that the cutoff is ignored outside MASK, so folding it in for the
// other modes would make two materials that draw alike compare as different.
const AlphaCoverageKey = struct {
    mode: res.MaterialInfo.Rendering.AlphaMode,
    cutoff: f32,
};

fn alphaCoverageKey(record: gpu.MaterialData) AlphaCoverageKey {
    const mode = record.alphaMode();
    return .{
        .mode = mode,
        .cutoff = if (mode == .mask) record.metallic_roughness_cutoff[2] else 0,
    };
}

// The other half of what makes two materials draw differently. A slot the mask
// calls absent is never sampled, so its transform is dead weight and two records
// differing only there are drawn alike; a mask that differs at all is observable
// on its own.
fn transformsDiffer(a: gpu.MaterialData, b: gpu.MaterialData) bool {
    for (std.enums.values(gpu.MaterialTextureSlot)) |slot| {
        const sampled = a.samplesSlot(slot);
        if (sampled != b.samplesSlot(slot)) return true;
        if (!sampled) continue;
        const index = @intFromEnum(slot);
        if (!std.meta.eql(a.tex[index], b.tex[index])) return true;
    }
    return false;
}

// What the sampler and KHR_texture_transform paths were given, said out loud at
// load time. All three are decided once per model and none changes per frame.
//
// It is the load-time partner the in-loop probes need: a run whose counts are
// all zero is a run in which the transform path was never exercised, and saying
// so is what stops a model with no transforms from reading as one that handles
// them.
fn reportTextureTransforms(
    model: *const gltf.importer.Model,
    packed_materials: []const gpu.MaterialData,
) void {
    const slots = @typeInfo(res.MaterialInfo.TextureMaps).@"struct".fields;

    var transformed: u32 = 0;
    var second_set: u32 = 0;
    var unmipmapped: u32 = 0;
    for (model.materials) |*material| {
        inline for (slots) |field| {
            const slot = @field(material.textures, field.name);
            if (slot.path != null) {
                if (!isIdentityTransform(slot.uv)) transformed += 1;
                if (slot.uv.set != 0) second_set += 1;
                if (slot.sampler.mipmap_mode == .none) unmipmapped += 1;
            }
        }
    }
    log.info(
        "texture transform: {d} slot(s) transformed, {d} on UV set 1, {d} sampled without mipmaps",
        .{ transformed, second_set, unmipmapped },
    );

    // The host packs what the importer read. A slot that lost its transform
    // between the two renders at the mesh's own UV and looks like an asset that
    // never asked for one, which is the failure this pass cannot see.
    var packed_transformed: u32 = 0;
    for (packed_materials) |record| {
        for (record.tex) |transform| {
            const identity = transform.rs[0] == 1 and transform.rs[1] == 0 and
                transform.rs[2] == 0 and transform.rs[3] == 1 and
                transform.params[0] == 0 and transform.params[1] == 0;
            if (!identity) packed_transformed += 1;
        }
    }
    var expected: u32 = 0;
    for (model.materials) |*material| {
        inline for (slots) |field| {
            if (!isIdentityTransform(@field(material.textures, field.name).uv)) expected += 1;
        }
    }
    check(
        packed_transformed == expected,
        "{d} of {d} transformed slot(s) reached the packed material array",
        .{ packed_transformed, expected },
    );

    // glTF 2.0 specification, 5.30.2: a mesh primitive must carry the texture
    // coordinate attribute a material's texCoord names. Nothing rejects an
    // asset that breaks it, and the shader's answer there is a slot sampling
    // zeros, so the run says which mesh it was rather than leaving a black
    // patch to guess at.
    for (model.meshes, 0..) |*mesh, index| {
        const material = &model.materials[mesh.material];
        var wants_second_set = false;
        inline for (slots) |field| {
            const slot = @field(material.textures, field.name);
            if (slot.path != null and slot.uv.set != 0) wants_second_set = true;
        }
        if (!wants_second_set) continue;
        check(
            mesh.streams.uv1,
            "mesh {d} carries TEXCOORD_1 for the material that reads UV set 1",
            .{index},
        );
    }

    // How many meshes reach the second-UV pipelines. Counted and not checked:
    // every variant has an entry point by the shape of the table the renderer
    // is built from, and the variant is a projection of the mesh's own streams,
    // so there is no pair here that can disagree. What a run of this can still
    // show is the count being zero on a model whose materials name TEXCOORD_1,
    // which is a stream lost between the parser and here.
    var uv1_meshes: u32 = 0;
    for (model.meshes) |*mesh| {
        if (gpu.sceneVariantFor(mesh.streams).uv1) uv1_meshes += 1;
    }
    log.info(
        "texture transform: {d} of {d} mesh(es) draw through a second-UV pipeline",
        .{ uv1_meshes, model.meshes.len },
    );
}

// The box the model occupies in normalized device coordinates, printed beside
// the viewport it will be stretched across.
fn reportNdcExtent(
    view_projection: zm.Mat,
    box: res.Aabb,
    aspect: f32,
    viewport: platform.Extent2D,
) void {
    var min = [2]f32{ std.math.inf(f32), std.math.inf(f32) };
    var max = [2]f32{ -std.math.inf(f32), -std.math.inf(f32) };
    for (aabbCorners(box)) |corner| {
        const clip = zm.mul(zm.f32x4(corner[0], corner[1], corner[2], 1), view_projection);
        if (!(clip[3] > 0)) continue;
        const ndc = [2]f32{ clip[0] / clip[3], clip[1] / clip[3] };
        min = .{ @min(min[0], ndc[0]), @min(min[1], ndc[1]) };
        max = .{ @max(max[0], ndc[0]), @max(max[1], ndc[1]) };
    }
    const width = max[0] - min[0];
    const height = max[1] - min[1];
    log.info(
        "ndc box: x [{d:.3} {d:.3}] y [{d:.3} {d:.3}], {d:.3} wide by {d:.3} tall, ratio {d:.3}",
        .{ min[0], max[0], min[1], max[1], width, height, height / width },
    );
    // The viewport turns normalized device coordinates into pixels, so a shape
    // that is square on screen is one whose device box has the viewport's
    // inverse ratio.
    const pixels_wide = width * 0.5 * @as(f32, @floatFromInt(viewport.width));
    const pixels_tall = height * 0.5 * @as(f32, @floatFromInt(viewport.height));
    log.info("on screen: {d:.0} by {d:.0} pixels, aspect {d:.4}, ratio {d:.3}", .{
        pixels_wide, pixels_tall, aspect, pixels_tall / pixels_wide,
    });
}

// Where the model's highest world point lands, in framebuffer rows.
//
// Vulkan specification, vkCmdSetViewport: with a positive height the viewport
// maps device y of -1 to the first row and +1 to the last, so a row near zero is
// the top of the window. The matrix passed in is the one the shader gets, flip
// included, which is what makes the answer the one on screen.
fn reportUpAxis(view_projection: zm.Mat, box: res.Aabb, viewport: platform.Extent2D) void {
    var highest = aabbCorners(box)[0];
    for (aabbCorners(box)) |corner| {
        if (corner[1] > highest[1]) highest = corner;
    }
    const clip = zm.mul(zm.f32x4(highest[0], highest[1], highest[2], 1), view_projection);
    const ndc_y = clip[1] / clip[3];
    const rows: f32 = @floatFromInt(viewport.height);
    const row = (ndc_y * 0.5 + 0.5) * rows;

    log.info("up axis: the highest point of the model lands at row {d:.0} of {d}", .{ row, viewport.height });
    log.info("up axis: {s}", .{
        if (row > rows * 0.5)
            "in the lower half, so the picture is upside down and the Y flip is unplaced"
        else
            "in the upper half, so the picture is the right way up",
    });
}
