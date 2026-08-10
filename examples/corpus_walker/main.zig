const std = @import("std");
const gltf = @import("lenore-gltf");
const gpu = @import("lenore-gpu");
const ktx = @import("lenore-ktx");
const lenore = @import("lenore");
const platform = @import("lenore-platform");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");
const zignal = @import("zignal");

const Allocator = std.mem.Allocator;
const log = std.log.scoped(.walker);

// A viewer over a directory of glTF assets, one at a time.
//
// It exists to be looked through. The validation app answers whether a stage
// agrees with the host; this answers what the corpus looks like, which is the
// question no check can be written for. So it carries no checks at all: what it
// has instead is a camera you can put anywhere, controls for the things that
// change a judgement, and a report of what a frame costs.
//
// Usage: run-corpus_walker -- <assets-root> [environment/] [--start=<name>]
//                            [--present=fifo|mailbox|immediate]
//                            [--compress] [--cache=<dir>]
//
// The present mode is here because a rate measured under `fifo` is the
// display's and not the renderer's: it paces every frame to the refresh and a
// scene that could run at three times it reports the same number as one that
// barely keeps up. `immediate` is what says which of the two a model is.
//
// The root is a directory of model directories, which is what the Khronos
// sample repository's `Models/` is. Each is opened through its `glTF-Binary`
// variant where it has one and its `glTF` variant otherwise, because those two
// are the forms this engine reads.

const checking = std.debug.runtime_safety;
var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

// Shadows are off to begin with, which is what the conformance corpus wants:
// geometric self-shadow has no equivalent on the flat impostors much of it is
// built from, and it contaminates a judgement about materials. `t` turns them
// on for the assets where the shadow is the point.
const initial_shadows = false;

// The device rings and every level's plan are sized from this one value. They
// are two capacities that have to agree: a level plans draws and joint palettes,
// and the rings are what those get written into, so a level planned larger than
// the ring fails at the write rather than at the plan. Written twice with
// different numbers, it fails on whichever model first exceeds the smaller.
//
// Measured over the corpus rather than chosen. The worst joint count is
// BrainStem at 1062, whose single mesh carries fifty-nine primitives that all
// share one eighteen-joint skin, and the worst draw count is
// NodePerformanceTest at ten thousand. Both are counted before the importer
// merges primitives, so both are upper bounds. An instance is 80 bytes and a
// joint 64, pinned by `lenore-gpu`'s own frame-set tests, so over two frames in
// flight this is 1.6 MB of instance ring and 0.26 MB of joint ring.
const walk_capacity: gpu.FrameCapacity = .{ .instances = 10240, .joints = 2048 };

// The largest material array a level may carry, measured the same way and over
// the same corpus. NodePerformanceTest declares ten thousand materials, the two
// Iridescence sphere grids three hundred and forty-four each, and nothing else
// passes one hundred and sixty-seven. One shared buffer rather than one per
// frame, at 224 bytes a record, so this is 2.2 MB.
const walk_materials: u32 = 10240;

// The largest artifact the cache will read back. A 4096 square with a chain is
// twenty-two megabytes, so this is past anything the corpus holds and short of
// a length that would be a denial of service by itself.
const max_artifact_bytes: usize = 128 << 20;

// The sun, in the angles the controls move rather than as a vector, because
// what a key does is turn it.
const Sun = struct {
    // From +X toward +Z, so the sun swings around the model.
    azimuth: f32 = 2.2,
    // Above the horizon. Never at the pole: straight down is where a fit has no
    // sideways axis to build from, and where every shadow hides under its
    // caster.
    elevation: f32 = 0.85,
    intensity: f32 = std.math.pi,

    const step = 0.05;
    const min_elevation = 0.05;
    const max_elevation = std.math.pi / 2.0 - 0.05;

    // The way the light travels, which is the direction a light carries. The
    // angles describe where it is, so this points back from there.
    fn travel(self: Sun) res.Vec3 {
        const horizontal = @cos(self.elevation);
        return .{
            -horizontal * @cos(self.azimuth),
            -@sin(self.elevation),
            -horizontal * @sin(self.azimuth),
        };
    }

    fn turn(self: *Sun, azimuth: f32, elevation: f32) void {
        self.azimuth += azimuth;
        self.elevation = std.math.clamp(
            self.elevation + elevation,
            min_elevation,
            max_elevation,
        );
    }
};

// One asset the walk can open.
const Entry = struct {
    // The model directory's own name, which is what the corpus calls the asset.
    name: []const u8,
    // Relative to the assets root, and the directory the loader is opened
    // against, so a document's own references cannot escape it.
    directory: []const u8,
    file: []const u8,

    fn deinit(self: *Entry, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.directory);
        allocator.free(self.file);
    }
};

// Which variant of a model to prefer, in order. Binary first because it is one
// file and needs no external reads; the plain form is what a third of the corpus
// ships instead.
const variants = [_][]const u8{ "glTF-Binary", "glTF" };

// Every model directory under the root, sorted, with the variant each will be
// opened through.
//
// A directory carrying neither variant is skipped rather than refused: the
// corpus holds forms this engine does not read, and a walk that stopped at the
// first of them would never reach the ones it does.
fn collect(allocator: Allocator, io: std.Io, root: std.Io.Dir) ![]Entry {
    var entries: std.ArrayList(Entry) = .empty;
    errdefer {
        for (entries.items) |*entry| entry.deinit(allocator);
        entries.deinit(allocator);
    }

    var models = try root.openDir(io, "Models", .{ .iterate = true });
    defer models.close(io);

    var walk = models.iterate();
    while (try walk.next(io)) |model| {
        if (model.kind != .directory) continue;

        for (variants) |variant| {
            const directory = try std.fmt.allocPrint(
                allocator,
                "Models/{s}/{s}",
                .{ model.name, variant },
            );
            const file = findDocument(allocator, io, root, directory) catch |err| {
                allocator.free(directory);
                if (err == error.FileNotFound) continue;
                return err;
            } orelse {
                allocator.free(directory);
                continue;
            };

            try entries.append(allocator, .{
                .name = try allocator.dupe(u8, model.name),
                .directory = directory,
                .file = file,
            });
            break;
        }
    }

    const owned = try entries.toOwnedSlice(allocator);
    std.mem.sort(Entry, owned, {}, struct {
        fn lessThan(_: void, a: Entry, b: Entry) bool {
            return std.mem.order(u8, a.name, b.name) == .lt;
        }
    }.lessThan);
    return owned;
}

// The one document in a variant directory, by extension. A variant holds its
// model and its buffers side by side, so the extension is what tells them apart.
fn findDocument(
    allocator: Allocator,
    io: std.Io,
    root: std.Io.Dir,
    directory: []const u8,
) !?[]const u8 {
    var dir = try root.openDir(io, directory, .{ .iterate = true });
    defer dir.close(io);

    var walk = dir.iterate();
    while (try walk.next(io)) |item| {
        if (item.kind != .file) continue;
        if (std.mem.endsWith(u8, item.name, ".glb") or std.mem.endsWith(u8, item.name, ".gltf"))
            return try allocator.dupe(u8, item.name);
    }
    return null;
}

// The model and everything resident for it. They are released together and in
// this order: the level borrows the model's meshes, so the document outlives it.
const Open = struct {
    loaded: gltf.loader.Loaded,
    model: gltf.importer.Model,
    level: lenore.Level,

    fn deinit(self: *Open, allocator: Allocator, engine: *lenore.Engine) void {
        engine.unload(&self.level);
        self.model.deinit(allocator);
        self.loaded.deinit(allocator);
    }
};

// A camera that goes where it is pushed.
//
// The engine's camera carries the pose and the projection; what a controller
// adds is the part that reads input, which is why this lives here. Framing a
// model puts it back on an orbit, and the first movement key drops the pivot:
// an orbit derives the eye from the pivot, so moving the eye directly is not
// expressible until it is detached.
const Freecam = struct {
    // World units a second at rest, set from the model's own radius when one is
    // opened. A corpus asset may be a metre across or a hundred, and a constant
    // that suits either crawls or overshoots on the other; a radius a second
    // crosses any of them in the same time.
    base_speed: f32 = 1.0,
    // The wheel's multiplier on it, kept across models so a chosen pace
    // survives a step. Geometric per notch: the useful range spans decades, and
    // a linear step is either imperceptible at the bottom of one or unusable at
    // the top.
    scale: f32 = 1.0,

    // One flag per direction rather than one signed value per axis. Two keys
    // name the vertical axis, and with a signed value releasing either would
    // stop the axis while the other was still held.
    held: struct {
        forward: bool = false,
        back: bool = false,
        left: bool = false,
        right: bool = false,
        up: bool = false,
        down: bool = false,
    } = .{},

    looking: bool = false,
    last_cursor: ?[2]f32 = null,

    const sensitivity: f32 = 0.0025;
    const scale_step: f32 = 1.25;
    const min_scale: f32 = 1.0 / 64.0;
    const max_scale: f32 = 64.0;

    fn axis(positive: bool, negative: bool) f32 {
        return @as(f32, if (positive) 1 else 0) - @as(f32, if (negative) 1 else 0);
    }

    fn advance(self: *const Freecam, camera: *scene.Camera, delta: f32) void {
        const forward = axis(self.held.forward, self.held.back);
        const strafe = axis(self.held.right, self.held.left);
        const rise = axis(self.held.up, self.held.down);
        // Opposed keys cancel, and a cancelled input must not reach the camera:
        // the write below detaches the orbit, which is not undone by moving by
        // nothing.
        if (forward == 0 and strafe == 0 and rise == 0) return;

        const placement = camera.placement();
        const speed = self.base_speed * self.scale * delta;
        // Up is the world's, not the camera's: a freecam that rises along its
        // own up drifts sideways whenever it is looking anywhere but level.
        const step = placement.front * @as(res.Vec3, @splat(forward * speed)) +
            placement.right * @as(res.Vec3, @splat(strafe * speed)) +
            res.Vec3{ 0, rise * speed, 0 };

        // Detached first. An orbit places the eye from the pivot, so writing an
        // eye into one changes nothing.
        camera.detach();
        camera.anchor = .{ .eye = camera.placement().position + step };
    }

    fn look(self: *Freecam, camera: *scene.Camera, position: [2]f32) void {
        defer self.last_cursor = position;
        if (!self.looking) return;
        const previous = self.last_cursor orelse return;

        camera.detach();
        camera.yaw += (position[0] - previous[0]) * sensitivity;
        camera.pitch = std.math.clamp(
            camera.pitch - (position[1] - previous[1]) * sensitivity,
            -std.math.pi / 2.0 + 0.01,
            std.math.pi / 2.0 - 0.01,
        );
    }

    // The wheel sets the pace. `lines` arrives already summed when the queue
    // coalesces a burst, so raising the step to it makes one flick worth the
    // notches it contained.
    fn wheel(self: *Freecam, lines: f32) void {
        if (lines == 0) return;
        self.scale = std.math.clamp(
            self.scale * std.math.pow(f32, scale_step, lines),
            min_scale,
            max_scale,
        );
    }

    fn key(self: *Freecam, physical: platform.PhysicalKey, pressed: bool) bool {
        switch (physical) {
            .w => self.held.forward = pressed,
            .s => self.held.back = pressed,
            .d => self.held.right = pressed,
            .a => self.held.left = pressed,
            // Both pairs move along world Y, so neither depends on where the
            // camera is looking.
            .space, .e => self.held.up = pressed,
            .shift_left, .shift_right, .q => self.held.down = pressed,
            else => return false,
        }
        return true;
    }
};

// The camera keeps whatever pose it was left in across a resize and across a
// model. Reframing on either would throw away wherever the operator had flown
// to, which is the one thing this application exists to let them do; `f` is what
// asks for the framing back.
const Walker = struct {
    entries: []Entry,
    index: usize,
    // Why the frame loop is to end, set in the event drain and read once it has
    // ended. A level may not be replaced inside the drain: unloading one the
    // frame in flight is reading is what that would do. The engine records no
    // reason of its own, so `.none` is the window having been closed.
    leaving: Leaving = .none,

    open: *Open,

    camera: Freecam = .{},
    sun: Sun = .{},
    lights: [gpu.max_lights]gpu.LightUniform = undefined,
    sun_moved: bool = false,

    shadows: bool = initial_shadows,
    bloom: bool = true,
    // How much of each bloom level carries into the one above it. Held here
    // rather than left at the default because what it sets is how wide a halo
    // reads, which only a comparison of several values against a reference
    // picture answers.
    bloom_scatter: f32 = (gpu.BloomSettings{}).scatter,
    // Whether the environment is drawn behind the model. Separate from whether
    // one is loaded: the two halves of what an environment costs are the
    // full-screen draw and the per-fragment image-based term, and turning the
    // draw off while the cubes stay bound is what tells them apart.
    background: bool = true,
    reported_windows: u64 = 0,

    const Leaving = enum { none, quit, previous, next };

    pub fn onEvent(self: *Walker, engine: *lenore.Engine, event: platform.Event) !void {
        switch (event.payload) {
            .cursor => |cursor| self.camera.look(&engine.camera, cursor.logical_position),
            .scroll => |wheel| self.camera.wheel(wheel.line_delta[1]),
            .mouse_button => |button| if (button.button == .left) {
                self.camera.looking = button.action == .press;
                if (!self.camera.looking) self.camera.last_cursor = null;
                // Pointer capture, which hides the cursor and unbounds its
                // position, so looking is not stopped by the window edge. A
                // compositor may refuse it; the drag still works from the
                // deltas either way, which is why this is not fatal.
                engine.window.setCursorMode(
                    if (self.camera.looking) .disabled else .normal,
                ) catch |err| log.warn("cursor capture unavailable: {t}", .{err});
            },
            .key => |key| {
                if (self.camera.key(key.physical, key.action != .release)) return;
                switch (key.action) {
                    .press => try self.command(engine, key.physical),
                    // Held keys repeat only where holding one means something.
                    // A toggle that repeated would flicker at the key repeat
                    // rate instead of switching.
                    .repeat => if (repeatable(key.physical)) try self.command(engine, key.physical),
                    .release => {},
                }
            },
            else => {},
        }
    }

    fn command(self: *Walker, engine: *lenore.Engine, physical: platform.PhysicalKey) !void {
        switch (physical) {
            .left_bracket => self.leave(engine, .previous),
            .right_bracket => self.leave(engine, .next),
            .f => _ = engine.frameCamera(self.open.level.world.sphere),
            .t => {
                self.shadows = !self.shadows;
                log.info("sun shadows {s}", .{if (self.shadows) "on" else "off"});
            },
            .b => {
                self.bloom = !self.bloom;
                log.info("bloom {s}", .{if (self.bloom) "on" else "off"});
            },
            .g => {
                self.background = !self.background;
                log.info("background {s}", .{if (self.background) "on" else "off"});
            },
            .p => {
                engine.paused = !engine.paused;
                log.info("rendering {s}", .{if (engine.paused) "paused" else "resumed"});
            },
            .j => self.turnSun(-Sun.step, 0),
            .l => self.turnSun(Sun.step, 0),
            .i => self.turnSun(0, Sun.step),
            .k => self.turnSun(0, -Sun.step),
            .n => {
                const count = self.open.level.world.clipCount();
                if (count == 0) return;
                const next: u16 = @intCast((@as(usize, self.open.level.world.activeClip() orelse 0) + 1) % count);
                _ = try self.open.level.world.playClip(next);
                // Counted from one against the count, which is what "of" reads
                // as. The index the world takes stays zero-based.
                log.info("clip {d} of {d}", .{ @as(usize, next) + 1, count });
            },
            .escape => self.leave(engine, .quit),
            else => {},
        }
    }

    // Ends the frame loop and records why. Stepping to another model is only
    // expressible from outside the loop, because the loop is what reads the
    // level, so a bracket key leaves it exactly as escape does and the two are
    // told apart afterwards.
    fn leave(self: *Walker, engine: *lenore.Engine, reason: Leaving) void {
        self.leaving = reason;
        engine.requestExit();
    }

    // Takes a newly opened model as the current one, and scales the camera to
    // it. Both halves in one place, because a walk that updated one and not the
    // other flies at the previous model's scale.
    fn adopt(self: *Walker, open: *Open) void {
        self.open = open;
        self.camera.base_speed = @max(open.level.world.sphere.radius, 0.01);
        self.sun_moved = true;
    }

    fn turnSun(self: *Walker, azimuth: f32, elevation: f32) void {
        self.sun.turn(azimuth, elevation);
        self.sun_moved = true;
    }

    // The camera keeps whatever pose it was left in. Reframing here would throw
    // away wherever the operator had flown to, which is the one thing this
    // application exists to let them do; `f` asks for the framing back.
    pub fn onResize(_: *Walker, _: *lenore.Engine, _: platform.Extent2D) !void {}
    pub fn onCompute(_: *Walker, _: *lenore.Engine, _: *lenore.Level, _: gpu.vk.CommandBuffer) !void {}

    pub fn onRecord(_: *Walker, _: *lenore.Engine, _: *lenore.Level, _: gpu.vk.CommandBuffer) !void {}
    pub fn onFrame(
        self: *Walker,
        engine: *lenore.Engine,
        level: *lenore.Level,
        time: lenore.FrameTime,
    ) !void {
        self.camera.advance(&engine.camera, time.delta);

        // The look is read at the top of every frame, so writing it here is what
        // the next one draws.
        engine.look.sun_shadow.enabled = self.shadows;
        engine.look.bloom = if (self.bloom) .{ .scatter = self.bloom_scatter } else null;
        engine.look.background = if (self.background and level.hasEnvironment())
            .environment
        else
            .clear;

        if (self.sun_moved) {
            self.sun_moved = false;
            self.lights[0] = lenore.packLight(try .directional(
                .{ 1, 1, 1 },
                self.sun.intensity,
                self.sun.travel(),
            ));
            // Cheap, and it decides for itself whether the map has drifted far
            // enough to be worth re-baking.
            engine.refitSun(level.world.sphere);
        }

        if (engine.fps_windows != self.reported_windows) {
            self.reported_windows = engine.fps_windows;
            if (engine.last_fps) |report| self.reportFrame(engine, report);
        }
    }

    fn reportFrame(
        self: *Walker,
        engine: *lenore.Engine,
        report: lenore.FpsCounter.Report,
    ) void {
        const phases = engine.last_phases;
        // The target extent belongs beside the rate. A compositor decides what a
        // window's framebuffer measures, and on a scaled output that is not
        // what was asked for; a millisecond count read without it says nothing
        // about how much work a frame was.
        const target = engine.renderer.targetExtent();
        log.info(
            "{s}: {d:.1} fps, {d:.2} ms mean, {d:.2} ms worst, {d}x{d}",
            .{
                self.entries[self.index].name,
                report.fps,
                report.mean_ms,
                report.worst_ms,
                target.width,
                target.height,
            },
        );
        // The host's own time, phase by phase, against the frame it sits in. The
        // two agree: every interval of the frame is in one of these, so a
        // difference between them is a phase that is not being measured. Where
        // the time lands says what the frame is waiting for, and under `fifo`
        // most of it is the display: a large `wait` or `events` beside small
        // everything else is a frame with time to spare.
        log.info(
            "  host {d:.3} of {d:.3} ms: events {d:.3}, wait {d:.3}, acquire {d:.3}, world {d:.3}, rings {d:.3}, record {d:.3}, submit {d:.3}, present {d:.3}",
            .{
                milliseconds(phases.total()),
                report.mean_ms,
                milliseconds(phases.events_ns),
                milliseconds(phases.wait_ns),
                milliseconds(phases.acquire_ns),
                milliseconds(phases.world_ns),
                milliseconds(phases.update_ns),
                milliseconds(phases.record_ns),
                milliseconds(phases.submit_ns),
                milliseconds(phases.present_ns),
            },
        );
    }
};

// Which keys mean something when held down.
fn repeatable(physical: platform.PhysicalKey) bool {
    return switch (physical) {
        .j, .l, .i, .k, .left_bracket, .right_bracket => true,
        else => false,
    };
}

fn milliseconds(nanoseconds: u64) f64 {
    return lenore.seconds(nanoseconds) * 1000.0;
}

pub fn main(process: std.process.Init.Minimal) !void {
    const gpa = if (checking) debug_allocator.allocator() else std.heap.smp_allocator;
    defer if (checking) {
        if (debug_allocator.deinit() == .leak) log.err("host memory leaked", .{});
    };

    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var arguments: std.process.Args.Iterator = .init(process.args);
    _ = arguments.skip();

    var root_path: ?[]const u8 = null;
    var environment_path: ?[]const u8 = null;
    var start_name: ?[]const u8 = null;
    var present: gpu.PresentModePreference = .fifo;
    var compress = false;
    var cache_path: ?[]const u8 = null;
    var bloom_scatter = (gpu.BloomSettings{}).scatter;
    while (arguments.next()) |argument| {
        if (std.mem.startsWith(u8, argument, "--start=")) {
            start_name = argument["--start=".len..];
            continue;
        }
        if (std.mem.eql(u8, argument, "--compress")) {
            compress = true;
            continue;
        }
        if (std.mem.startsWith(u8, argument, "--cache=")) {
            cache_path = argument["--cache=".len..];
            continue;
        }
        if (std.mem.startsWith(u8, argument, "--bloom-scatter=")) {
            const value = argument["--bloom-scatter=".len..];
            bloom_scatter = std.fmt.parseFloat(f32, value) catch {
                log.err("--bloom-scatter takes a number, got '{s}'", .{value});
                return error.InvalidArgument;
            };
            // The range `BloomSettings.resolve` accepts. Checked here as well
            // because that one runs inside a frame, where its error would
            // surface as a dead run rather than as a bad argument.
            if (!(bloom_scatter >= 0 and bloom_scatter < 1)) {
                log.err("--bloom-scatter takes 0 <= s < 1, got '{s}'", .{value});
                return error.InvalidArgument;
            }
            continue;
        }
        if (std.mem.startsWith(u8, argument, "--present=")) {
            const value = argument["--present=".len..];
            present = std.meta.stringToEnum(gpu.PresentModePreference, value) orelse {
                log.err("--present takes fifo, mailbox or immediate, got '{s}'", .{value});
                return error.InvalidArgument;
            };
            continue;
        }
        if (root_path == null) {
            root_path = argument;
        } else if (environment_path == null) {
            environment_path = argument;
        } else {
            log.err("usage: corpus_walker <assets-root> [environment/] [--start=<name>]", .{});
            return error.TooManyArguments;
        }
    }
    const assets_root = root_path orelse {
        log.err("usage: corpus_walker <assets-root> [environment/] [--start=<name>]", .{});
        return error.MissingAssetsRoot;
    };

    var root = try std.Io.Dir.cwd().openDir(io, assets_root, .{});
    defer root.close(io);

    var cache: ?std.Io.Dir = null;
    defer if (cache) |*dir| dir.close(io);
    if (cache_path) |path| {
        cache = try std.Io.Dir.cwd().createDirPathOpen(io, path, .{});
        log.info("cache: {s}", .{path});
    }

    const entries = try collect(gpa, io, root);
    defer {
        for (entries) |*entry| entry.deinit(gpa);
        gpa.free(entries);
    }
    if (entries.len == 0) return error.NoAssetsFound;
    log.info("corpus: {d} model(s) under {s}", .{ entries.len, assets_root });

    var index: usize = 0;
    if (start_name) |wanted| {
        index = for (entries, 0..) |entry, position| {
            if (std.mem.eql(u8, entry.name, wanted)) break position;
        } else {
            log.err("no model named '{s}' under {s}", .{ wanted, assets_root });
            return error.NoSuchModel;
        };
    }

    var engine: lenore.Engine = undefined;
    try engine.init(gpa, .{
        .title = "Lenore corpus walker",
        .frame_capacity = walk_capacity,
        .material_capacity = walk_materials,
        .present = present,
    });
    defer engine.deinit();
    log.info("device: {s}", .{engine.context.deviceName()});

    // Reported because a screenshot taken to compare halo widths has to carry
    // which value produced it.
    log.info("bloom scatter: {d}", .{bloom_scatter});

    var walker: Walker = .{
        .entries = entries,
        .index = index,
        .open = undefined,
        .bloom_scatter = bloom_scatter,
    };
    walker.lights[0] = lenore.packLight(try .directional(
        .{ 1, 1, 1 },
        walker.sun.intensity,
        walker.sun.travel(),
    ));
    engine.look = .{
        .lights = walker.lights[0..1],
        .sun_shadow = .{ .enabled = initial_shadows, .strength = 1, .normal_offset_texels = 1 },
    };

    const corpus: Corpus = .{
        .root = root,
        .environment_path = environment_path,
        .compress = compress,
        .cache = cache,
    };

    var failures: std.ArrayList(Failure) = .empty;
    defer failures.deinit(gpa);
    const visited = try gpa.alloc(bool, entries.len);
    defer gpa.free(visited);
    @memset(visited, false);
    // Runs before the list is freed, and on every way out including escape and
    // the window closing, which is how a partial walk still reports what it met.
    defer reportFailures(failures.items, visited);

    var open = (try openSkipping(gpa, io, &engine, corpus, &walker, .forward, &failures, visited)) orelse
        return error.NoModelOpened;
    // Whether `open` still holds anything. The loop releases it before opening
    // the next model, so every exit below has to know which side of that it is
    // on; a `defer` alone would release it a second time on the paths that
    // already have.
    var resident = true;
    defer if (resident) open.deinit(gpa, &engine);
    walker.adopt(&open);

    logControls();
    while (true) {
        engine.run(&open.level, &walker) catch |err| {
            log.err("{s}: {t}", .{ entries[walker.index].name, err });
            return err;
        };

        const leaving = walker.leaving;
        walker.leaving = .none;

        // The old level is released before the new one is opened, so a walk
        // never holds two models resident. Everything the device kept of it is
        // given back by `unload`, which the engine sequences.
        open.deinit(gpa, &engine);
        resident = false;

        const direction: Direction = switch (leaving) {
            .none, .quit => return,
            .next => .forward,
            .previous => .backward,
        };
        walker.index = direction.advance(walker.index, entries.len);

        open = (try openSkipping(gpa, io, &engine, corpus, &walker, direction, &failures, visited)) orelse
            return error.NoModelOpened;
        resident = true;
        walker.adopt(&open);
    }
}

// What every open reads and no open changes. Bundled because the walk opens
// from two places and threads all four through a skip loop besides.
const Corpus = struct {
    root: std.Io.Dir,
    environment_path: ?[]const u8,
    compress: bool,
    cache: ?std.Io.Dir,
};

// A model this engine would not read, kept for the report at the end. The name
// is borrowed from the entry list, which outlives every walk over it.
const Failure = struct {
    name: []const u8,
    err: anyerror,
};

const Direction = enum {
    forward,
    backward,

    fn advance(self: Direction, index: usize, count: usize) usize {
        return switch (self) {
            .forward => (index + 1) % count,
            .backward => (index + count - 1) % count,
        };
    }
};

// Opens the entry the walker is on, and on refusal walks on by `direction`
// until one opens. The corpus holds encodings this engine does not read, and a
// walk that stopped at the first of them would never reach the models past it,
// which is the whole of what the walk is for. Null means every entry refused.
fn openSkipping(
    allocator: Allocator,
    io: std.Io,
    engine: *lenore.Engine,
    corpus: Corpus,
    walker: *Walker,
    direction: Direction,
    failures: *std.ArrayList(Failure),
    visited: []bool,
) !?Open {
    const count = walker.entries.len;
    for (0..count) |_| {
        const entry = walker.entries[walker.index];
        visited[walker.index] = true;
        if (openModel(allocator, io, engine, corpus, entry)) |open| {
            return open;
        } else |err| {
            log.err("{s}: {t}", .{ entry.name, err });
            // Recorded once however often the walk passes it. Stepping back and
            // forth over a model that refuses is ordinary, and a report that
            // counted every pass would say nothing about the corpus.
            if (!recorded(failures.items, entry.name))
                try failures.append(allocator, .{ .name = entry.name, .err = err });
            walker.index = direction.advance(walker.index, count);
        }
    }
    return null;
}

fn recorded(failures: []const Failure, name: []const u8) bool {
    for (failures) |failure| {
        if (std.mem.eql(u8, failure.name, name)) return true;
    }
    return false;
}

// Which of the corpus this engine refuses, and why. The most useful single
// artifact of a walk, so it is printed on every way out rather than only on the
// paths that finish.
//
// Counted against what the walk actually reached and never against the corpus
// size. A walk that opened one model and quit is not evidence about the other
// hundred and forty-seven, and a report that said so would be the most
// confidently wrong line the program prints.
fn reportFailures(failures: []const Failure, visited: []const bool) void {
    var seen: usize = 0;
    for (visited) |one| {
        if (one) seen += 1;
    }
    if (failures.len == 0) {
        log.info("{d} of {d} model(s) visited, none refused", .{ seen, visited.len });
        return;
    }
    log.info(
        "{d} of {d} model(s) visited, {d} refused:",
        .{ seen, visited.len, failures.len },
    );
    for (failures) |failure| log.info("  {s}: {t}", .{ failure.name, failure.err });
}

fn openModel(
    allocator: Allocator,
    io: std.Io,
    engine: *lenore.Engine,
    corpus: Corpus,
    entry: Entry,
) !Open {
    log.info("opening {s} ({s}/{s})", .{ entry.name, entry.directory, entry.file });

    var directory = try corpus.root.openDir(io, entry.directory, .{});
    defer directory.close(io);

    var loaded = try gltf.loader.open(allocator, io, directory, entry.file);
    errdefer loaded.deinit(allocator);
    var model = try gltf.importer.build(allocator, &loaded.document, loaded.directory);
    errdefer model.deinit(allocator);
    try gltf.importer.readExternalImages(allocator, io, directory, &model);
    if (corpus.compress) try compressImages(allocator, io, &model, corpus.cache);

    var level = try lenore.Level.init(allocator, engine.deps(), &model, .{
        .capacity = walk_capacity,
    });
    errdefer level.deinit(&engine.textures);

    if (corpus.environment_path) |path| try level.openEnvironment(engine.deps(), path);
    try engine.install(&level, &model);

    log.info("{s}: {d} mesh(es), {d} material(s), radius {d:.3}", .{
        entry.name,
        model.meshes.len,
        model.materials.len,
        level.world.sphere.radius,
    });
    return .{ .loaded = loaded, .model = model, .level = level };
}

// Which format a slot demands of a KTX2 image, as `MaterialSlot.format` maps
// it: base colour and emissive are read as colour and stored sRGB, and the
// three that carry numbers are stored linear. This is not a preference. A file
// whose header says the other one is refused on upload with FormatMismatch,
// because the sampler would undo a transfer that was never applied.
fn semanticOf(slot: []const u8) ktx.Semantic {
    if (std.mem.eql(u8, slot, "base_colour") or std.mem.eql(u8, slot, "emissive"))
        return .colour;
    return .data;
}

// What each image has to be converted as, or null where it cannot be.
//
// An image named by two slots of different meaning has no single answer: one
// artifact carries one format code, and whichever is written refuses the other
// slot on upload. Such an image is left as it is rather than converted wrongly.
// It is rare and worth reporting when it happens; it is not worth a second copy
// of the image on the device.
const Claim = enum {
    unclaimed,
    colour,
    data,
    conflicting,

    fn with(self: Claim, wanted: ktx.Semantic) Claim {
        const asked: Claim = switch (wanted) {
            .colour => .colour,
            .data => .data,
        };
        return switch (self) {
            .unclaimed => asked,
            .conflicting => .conflicting,
            else => if (self == asked) self else .conflicting,
        };
    }

    fn semantic(self: Claim) ?ktx.Semantic {
        return switch (self) {
            .colour => .colour,
            .data => .data,
            .unclaimed, .conflicting => null,
        };
    }
};

fn imageSemantics(
    allocator: Allocator,
    model: *const gltf.importer.Model,
) ![]?ktx.Semantic {
    const claims = try allocator.alloc(Claim, model.images.len);
    defer allocator.free(claims);
    @memset(claims, .unclaimed);

    const slots = @typeInfo(res.MaterialInfo.TextureMaps).@"struct".fields;
    for (model.materials) |*material| {
        inline for (slots) |field| {
            const reference = @field(material.textures, field.name);
            if (reference.path) |key| {
                for (model.images, claims) |*image, *claim| {
                    if (std.mem.eql(u8, image.key, key))
                        claim.* = claim.with(semanticOf(field.name));
                }
            }
        }
    }

    const resolved = try allocator.alloc(?ktx.Semantic, model.images.len);
    for (claims, resolved) |claim, *out| out.* = claim.semantic();
    return resolved;
}

// Replaces every image the model carries with a BC7 artifact.
//
// The bytes belong to the model, so the converted ones are put back where the
// source ones were and freed with it. An image the decoder cannot read, or one
// with no single answer to what it is, is left as it is rather than failing the
// model: the corpus holds encodings this engine does not decode, and a walk
// that stopped at the first of them would never reach the ones it does.
// What a converted image is filed under.
//
// The content and nothing else: the source bytes and what they are being
// converted as. Two models sharing a texture share the artifact, a model
// reconverted after an edit misses, and nothing has to be invalidated by hand.
// The semantic is part of the key because the same pixels stored as colour and
// as data are two different files.
fn cacheName(bytes: []const u8, semantic: ktx.Semantic, out: *[70]u8) []const u8 {
    var digest: [32]u8 = undefined;
    var hash: std.crypto.hash.Blake3 = .init(.{});
    hash.update(bytes);
    hash.update(&[_]u8{@intFromEnum(semantic)});
    hash.final(&digest);
    return std.fmt.bufPrint(out, "{x}.ktx2", .{&digest}) catch unreachable;
}

fn compressImages(
    allocator: Allocator,
    io: std.Io,
    model: *gltf.importer.Model,
    cache: ?std.Io.Dir,
) !void {
    const workers = ktx.automaticWorkers();
    const semantics = try imageSemantics(allocator, model);
    defer allocator.free(semantics);

    var wanted: usize = 0;
    for (semantics) |semantic| wanted += @intFromBool(semantic != null);
    if (wanted == 0) return;

    var converted: usize = 0;
    var reused: usize = 0;
    var ambiguous: usize = 0;
    // What the device would otherwise hold, which is the only figure the
    // artifact is comparable against. The encoded source is a tenth of either
    // and says nothing about what a frame costs.
    var decoded_bytes: usize = 0;
    var artifact_bytes: usize = 0;

    log.info("compressing {d} image(s) to BC7 over {d} worker(s)", .{ wanted, workers });
    for (model.images, semantics) |*source, semantic| {
        const bytes = source.bytes orelse continue;
        // Already a device format, which is what a converted asset tree hands
        // over and what a second pass over one would otherwise decode.
        if (gpu.isKtx2(bytes)) continue;
        const wants = semantic orelse {
            if (source.bytes != null) ambiguous += 1;
            continue;
        };

        var name_buffer: [70]u8 = undefined;
        const name = cacheName(bytes, wants, &name_buffer);

        var from_cache = false;
        var artifact: []u8 = undefined;
        var extent: [2]u32 = .{ 0, 0 };

        if (cache) |dir| read: {
            artifact = dir.readFileAlloc(io, name, allocator, .limited(max_artifact_bytes)) catch
                break :read;
            from_cache = true;
        }

        if (!from_cache) {
            var decoded = zignal.Image(zignal.Rgba(u8)).loadFromBytes(allocator, bytes) catch |err| {
                log.warn("image {s} left as it is: {t}", .{ source.key, err });
                continue;
            };
            defer decoded.deinit(allocator);
            if (decoded.stride != decoded.cols) continue;
            extent = .{ @intCast(decoded.cols), @intCast(decoded.rows) };

            artifact = ktx.convert(
                allocator,
                io,
                std.mem.sliceAsBytes(decoded.data),
                extent[0],
                extent[1],
                .{ .semantic = wants, .workers = workers },
            ) catch |err| {
                log.warn("image {s} left as it is: {t}", .{ source.key, err });
                continue;
            };
            decoded_bytes += @as(usize, extent[0]) * extent[1] * 4;
            converted += 1;

            // Written after the conversion succeeded, so a failed one leaves
            // nothing behind for the next run to read as an artifact. A write
            // that fails costs a reconversion later and nothing else.
            if (cache) |dir| dir.writeFile(io, .{ .sub_path = name, .data = artifact }) catch |err|
                log.warn("cache write failed for {s}: {t}", .{ source.key, err });
        } else {
            reused += 1;
        }

        artifact_bytes += artifact.len;
        // Printed as it happens rather than summed at the end: a 2048 square is
        // most of a second, and a load that says nothing for half a minute
        // reads as one that has hung.
        if (from_cache) {
            log.info("  {d}/{d} {s}: {t} from cache, {d:.1} MB", .{
                converted + reused,
                wanted,
                source.key,
                wants,
                @as(f64, @floatFromInt(artifact.len)) / (1 << 20),
            });
        } else {
            log.info("  {d}/{d} {s}: {d}x{d} {t} to {d:.1} MB", .{
                converted + reused,
                wanted,
                source.key,
                extent[0],
                extent[1],
                wants,
                @as(f64, @floatFromInt(artifact.len)) / (1 << 20),
            });
        }

        allocator.free(bytes);
        source.bytes = artifact;
    }

    if (ambiguous > 0) {
        log.warn(
            "{d} image(s) are read as colour by one slot and as data by another, and were left uncompressed",
            .{ambiguous},
        );
    }
    if (converted + reused > 0) {
        // The artifact carries a mip chain and the uncompressed path does not,
        // so this is not a ratio between two of the same thing: it is four to
        // one per level, less a third given back to the levels that did not
        // exist before.
        log.info(
            "{d} image(s) converted and {d} reused: {d:.1} MB decoded to {d:.1} MB of BC7 with mips",
            .{
                converted,
                reused,
                @as(f64, @floatFromInt(decoded_bytes)) / (1 << 20),
                @as(f64, @floatFromInt(artifact_bytes)) / (1 << 20),
            },
        );
    }
}

fn logControls() void {
    log.info("controls: [ ] previous/next model, f frame, n cycle clip, p pause, escape quit", .{});
    log.info("          w a s d fly, space/shift up/down along Y (e/q too), wheel sets speed", .{});
    log.info("          hold left mouse to look", .{});
    log.info("          t sun shadows, b bloom, g background, i j k l move the sun", .{});
}
