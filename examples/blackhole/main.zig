const std = @import("std");
const gpu = @import("lenore-gpu");
const imui = @import("lenore-imui");
const lenore = @import("lenore");
const platform = @import("lenore-platform");
const OrbitControl = @import("example-orbit").OrbitControl;
const zm = @import("zmath");

const log = std.log.scoped(.blackhole);

// A Schwarzschild black hole, integrated per pixel inside the engine's main
// pass.
//
// The whole picture comes out of one fullscreen draw the driver records, so
// this application loads no asset and its level is empty. What it does use is
// everything after the main pass: the disk's brightness spans orders of
// magnitude by construction, and the bloom chain and the tone operator are what
// turn that into a picture rather than a white disc.
//
// The shader reads no descriptor set, so recording it costs a pipeline bind, a
// push and three vertices, and nothing the scene bound can be disturbed.

// Usage: run-blackhole [--tier=NAME] [--spin=-1|1]

const usage =
    \\usage: blackhole [--tier=interactive|standard|refined|still] [--spin=-1|1]
    \\
    \\  1..4          switch tier while running
    \\  left drag     orbit; release to throw
    \\  mouse wheel   smooth optical zoom toward the centre
    \\  r             reverse the disk
    \\  c             doppler colour on / off
    \\  - / =         exposure down / up
    \\  t / o         disk thickness / opacity
    \\  d             cycle the debug picture
    \\  escape        quit
;

// What the integration is allowed to spend, and the only thing that separates
// one tier from the next.
//
// The two numbers are not independent: the step count is a budget and the turn
// per step is what spends it, so a finer turn wants more steps to reach the
// same distance. The pairs below are chosen together.
//
// No frame time is claimed for any of them. What each costs on a given device
// is what running it reports, and the tier names say what they are for rather
// than what they achieve.
const Tier = struct {
    name: []const u8,
    max_steps: u32,
    max_turn: f32,

    // A ceiling on how far a ray can wind, not the reach it has.
    //
    // Every step turns through at most `max_turn`, so winding once costs at
    // least 2*pi/max_turn steps. It is only a ceiling because the steps taken
    // inside the disk are cut to a fraction of its thickness and turn through
    // far less than that, and they come out of the same budget: a ray crossing
    // the disk at a shallow angle spends most of its steps there and winds less
    // than this says. The photon ring is where the difference shows.
    fn windings(self: Tier) f32 {
        return @as(f32, @floatFromInt(self.max_steps)) * self.max_turn / std.math.tau;
    }
};

// The budgets differ in reach and not only in accuracy, which the first set of
// these did not: at 96/0.14, 320/0.045, 1200/0.012 and 6000/0.003 every tier
// came out between 2.1 and 2.9 windings, so no tier let a ray wind far enough
// to close the photon ring and the four differed in step size alone. The
// program printed that itself through `windings`.
const tiers = [_]Tier{
    // A ceiling of three windings, and in practice about one once the disk has
    // taken its share. Enough to bend a ray and carry it through the medium
    // twice, which is the disk's own lensed image over and under the shadow.
    .{ .name = "interactive", .max_steps = 160, .max_turn = 0.131 },
    // Six and a half. The ring appears here, and the step is still coarse
    // enough that its inner edge is where the integration error shows first.
    .{ .name = "standard", .max_steps = 640, .max_turn = 0.065 },
    // Twelve, at a step twice as fine as the tier above.
    .{ .name = "refined", .max_steps = 2400, .max_turn = 0.031 },
    // Twenty-four, and fine. For a still: the budget is past where more of it
    // moves the picture, and what it buys is that nothing in the frame is
    // limited by it.
    .{ .name = "still", .max_steps = 12000, .max_turn = 0.0126 },
};

const default_tier = 1;

// What each debug picture answers, in the order `d` cycles them. Named here so
// the log says which one is on rather than a number.
const debugNames = [_][]const u8{
    "render",
    "ending: red captured, green escaped, blue out of budget",
    "samples taken inside the medium, against 64",
    "budget spent",
};

// Mirrors `PushConstants` in blackhole.slang.
//
// The offsets are the compiler's, read from the reflection JSON slangc emits
// beside the words: 0, 16, 32, 48 for the four vectors and then eleven scalars
// from 64 to 116, for 120 bytes. The asserts below hold this declaration to them,
// so a field inserted here fails the build instead of shifting every value the
// shader reads after it.
//
// `extern` so the field order is the declaration order.
const Push = extern struct {
    ray_right: [4]f32,
    ray_up: [4]f32,
    ray_front: [4]f32,
    eye: [4]f32,

    max_steps: u32,
    max_turn: f32,
    disk_inner: f32,
    disk_outer: f32,
    disk_temperature: f32,
    disk_spin: f32,
    doppler_colour: u32,
    exposure: f32,
    star_intensity: f32,
    time: f32,
    turbulence: f32,
    disk_thickness: f32,
    disk_opacity: f32,
    debug: u32,
};

comptime {
    std.debug.assert(@offsetOf(Push, "ray_right") == 0);
    std.debug.assert(@offsetOf(Push, "ray_up") == 16);
    std.debug.assert(@offsetOf(Push, "ray_front") == 32);
    std.debug.assert(@offsetOf(Push, "eye") == 48);
    std.debug.assert(@offsetOf(Push, "max_steps") == 64);
    std.debug.assert(@offsetOf(Push, "max_turn") == 68);
    std.debug.assert(@offsetOf(Push, "disk_inner") == 72);
    std.debug.assert(@offsetOf(Push, "disk_outer") == 76);
    std.debug.assert(@offsetOf(Push, "disk_temperature") == 80);
    std.debug.assert(@offsetOf(Push, "disk_spin") == 84);
    std.debug.assert(@offsetOf(Push, "doppler_colour") == 88);
    std.debug.assert(@offsetOf(Push, "exposure") == 92);
    std.debug.assert(@offsetOf(Push, "star_intensity") == 96);
    std.debug.assert(@offsetOf(Push, "time") == 100);
    std.debug.assert(@offsetOf(Push, "turbulence") == 104);
    std.debug.assert(@offsetOf(Push, "disk_thickness") == 108);
    std.debug.assert(@offsetOf(Push, "disk_opacity") == 112);
    std.debug.assert(@offsetOf(Push, "debug") == 116);
    std.debug.assert(@sizeOf(Push) == 120);
}

// Both stages read the block. The vertex stage builds the ray from the basis
// and the eye, and the fragment stage reads all of it, so a range naming one
// stage would leave the other reading a block it was not given.
const push_range: gpu.vk.PushConstantRange = .{
    .stage_flags = .{ .vertex_bit = true, .fragment_bit = true },
    .offset = 0,
    .size = @sizeOf(Push),
};

// The vertices the shader generates from `SV_VertexID`. It reads no buffer, so
// the pipeline declares no vertex input.
const vertex_count: u32 = 3;

// `@embedFile` yields bytes and SPIR-V is words. Copied into an aligned
// constant so the reinterpretation is valid rather than merely likely: an
// embedded file has no alignment of its own.
fn words(comptime bytes: anytype) []const u32 {
    const aligned: [bytes.len]u8 align(@alignOf(u32)) = bytes;
    const count = aligned.len / @sizeOf(u32);
    comptime std.debug.assert(count * @sizeOf(u32) == aligned.len);
    return @as([*]const u32, @ptrCast(&aligned))[0..count];
}

// The one pipeline this application owns.
//
// Built against the main pass's formats, which are read from the renderer
// rather than named here: both are the first candidate the device's features
// carried, so neither is knowable before the device exists. A resize does not
// change them, so this outlives one.
//
// `.background` is the mode, and it is the correct one rather than a
// convenience. The draw sits at the far plane and tests depth without writing
// it, so it appears exactly where no opaque surface stands and a scene drawn
// alongside would occlude it.
// One module, one layout, one fullscreen pass. The table is the whole of what
// this example builds on the device.
const Spec = gpu.ShaderEffectSpec;
const LayoutConfig = gpu.PipelineLayoutConfig;
const Shaders = gpu.ShaderEffect(.{
    .modules = .{"blackhole"},
    .layouts = .{"blackhole"},
    .pipelines = .{
        .disk = Spec{ .module = "blackhole", .layout = "blackhole", .stage = .{ .graphics = .{
            .vertex = "vertexMain",
            .fragment = "fragmentMain",
            .mode = .background,
            .culling = .{ .fixed = .{} },
        } } },
    },
});

const Effect = struct {
    context: *const gpu.Context,
    shaders: Shaders,

    fn init(context: *const gpu.Context, formats: gpu.PipelineFormats) !Effect {
        return .{
            .context = context,
            .shaders = try .init(context, .{
                .modules = .{ .blackhole = words(@embedFile("blackhole").*) },
                .layouts = .{ .blackhole = LayoutConfig{ .push_constants = &.{push_range} } },
                .formats = formats,
            }),
        };
    }

    // Vulkan specification, vkDestroyPipeline and the rest: every submission
    // naming any of these must have completed. The caller drains the device.
    fn deinit(self: *Effect) void {
        self.shaders.deinit(self.context);
        self.* = undefined;
    }

    fn record(self: *const Effect, commands: gpu.vk.CommandBuffer, push: Push) void {
        const device = self.context.device;
        device.cmdBindPipeline(commands, .graphics, self.shaders.get(.disk));
        device.cmdPushConstants(
            commands,
            self.shaders.layoutFor(.disk),
            push_range.stage_flags,
            push_range.offset,
            push_range.size,
            @ptrCast(&push),
        );
        device.cmdDraw(commands, vertex_count, 1, 0, 0);
    }
};

// The disk, in the geometric units the shader works in: the hole's mass is one,
// so the horizon is at 2 and the innermost stable circular orbit at 6.
//
// The outer edge is a choice and the inner one is not.
//
// Fifty leaves a broad cool outer annulus around the hot centre. The emissive
// profile begins its final taper at r = 40 and reaches zero at this rim, so the
// boundary remains invisible while the disk itself extends across the frame.
const disk_inner: f32 = 6;
const disk_outer: f32 = 50;

// The half thickness as a fraction of the radius.
//
// Two things turn on it, and the second is not obvious. It is what makes this
// gas rather than a plane: at zero every ray meets the disk once and returns
// one sample, which is a textured surface however good the texture.
//
// It also decides whether the shadow stays dark. The eye sits at `default_pitch`
// above the plane, so a ray aimed at the hole travels at a constant
// tan(pitch) / thickness scale heights the whole way in. At 0.06 against a
// pitch of 0.14 that is 2.3 heights, where the Gaussian still leaves 0.066, and
// over the thirty-odd units of path it gathers a visible haze across the
// shadow. Halving the thickness doubles the height and the Gaussian answers
// exponentially.
const disk_thickness: f32 = 0.03;

// How opaque the fluid is along the path. Enough that a dense filament takes a
// visible bite out of what is behind it, which is where the dark lanes across a
// bright disk come from, and not so much that the far side stops showing
// through: an accretion disk is not a wall.
const disk_opacity: f32 = 0.9;

// The inner-edge parameter of the temperature profile. Not a temperature the
// disk reaches: the profile peaks at 0.488 of this, so the hottest annulus is
// near 10700 K. At r = 40, where the outer taper begins, it is about 4700 K.
// That range makes the outer disk warm and carries the centre into blue-white.
//
// A stellar-mass hole accreting near its limit peaks in soft X-rays, which has
// no visible colour. This scale is therefore an authored visible-light mapping,
// not a claim about the physical mass or accretion rate.
const disk_temperature: f32 = 22000;

// Outside the disk and far from the hole. The second is what the shader's own
// comment asks for, since the photon energy is taken as one and that is exact
// only for an eye at infinity, with an error of order 1/r here; the first is
// because an eye inside the annulus has the disk around it rather than in front
// of it.
//
// At this distance the disk spans about seventy degrees and the shadow about
// five, which is roughly the framing the published visualisations use.
const default_distance: f32 = 55;

// Just off the plane. Edge-on hides the disk behind its own inner edge and
// face-on removes the beaming asymmetry, which is the most physical thing in
// the picture.
const default_pitch: f32 = -0.14;

// Zoom changes the projection rather than moving the eye. The ray integrator
// assumes that the eye stays outside the disk, while narrowing the field of view
// can magnify the photon ring without violating that assumption. Multiplying
// tan(fov / 2) makes every wheel notch the same image-space magnification.
const zoom_step: f32 = 0.82;
const min_fov_y: f32 = 0.02;
const max_fov_y: f32 = std.math.pi / 2.0;
const zoom_response: f32 = 12;
const zoom_snap: f32 = 1e-4;

const Driver = struct {
    effect: *const Effect,
    tier: usize = default_tier,
    spin: f32 = 1,
    orbit: OrbitControl = .{},

    // Half-height of the requested projection at unit depth. Null means the
    // camera has reached it and no zoom work remains.
    zoom_target: ?f32 = null,

    // Where the unshifted peak of the disk lands, in the units the tone
    // operator renders one as white. Above one by design: the bloom chain
    // thresholds at one and reaches full contribution at three, so a disk whose
    // peak sits at one contributes almost nothing to the glow, and the beamed
    // edge is what should be carrying it.
    // Set so the beamed edge clips and the rest of the disk grades. The two
    // edges differ by about twenty-five times at the profile's peak, which is
    // the Doppler shift and is correct, so no exposure renders both ends inside
    // the operator's range: this one puts the receding side low and lets the
    // approaching side carry the glow.
    exposure: f32 = 0.55,
    star_intensity: f32 = 1.6,

    // How far the temperature swings. A fifth, because the band response is
    // exponential in the reciprocal temperature: this is already several times
    // in brightness, and more would be noise rather than structure.
    turbulence: f32 = 0.2,

    // The medium's own two numbers, on keys because they are what a run is for.
    thickness: f32 = disk_thickness,
    opacity: f32 = disk_opacity,

    // Whether the palette follows the shifted temperature. On, because the
    // colour split across the inner edge is the physical statement; off is the
    // authored look, kept on a key because the two are worth comparing side by
    // side and the difference is the whole point of the control.
    doppler_colour: bool = true,

    // Accumulated from the frame deltas rather than read from the clock. The
    // hook that carries a time runs after the one that records, so this is the
    // previous frame's total, one frame stale and invisible at any rate the
    // disk turns at.
    elapsed: f32 = 0,

    // Which picture the fragment stage returns. Zero is the render.
    debug: u32 = 0,

    // The last window this reported. The report's own numbers are not an
    // identity, since two windows can measure the same duration.
    reported_windows: u64 = 0,

    pub fn onEvent(self: *Driver, engine: *lenore.Engine, event: platform.Event, _: bool) !void {
        switch (event.payload) {
            .cursor => |cursor| self.orbit.dragTo(
                &engine.camera,
                cursor.logical_position,
                event.timestamp_ns,
            ),
            .mouse_button => |button| if (button.button == .left) switch (button.action) {
                .press => {
                    self.orbit.begin(button.logical_position, event.timestamp_ns);
                    engine.window.setCursorMode(.disabled) catch |err|
                        log.warn("cursor capture unavailable: {t}", .{err});
                },
                .release => {
                    self.orbit.end(
                        &engine.camera,
                        button.logical_position,
                        event.timestamp_ns,
                    );
                    engine.window.setCursorMode(.normal) catch |err|
                        log.warn("cursor release unavailable: {t}", .{err});
                },
                .repeat => {},
            },
            .focus => |focus| if (!focus.focused) {
                self.orbit.cancel();
                engine.window.setCursorMode(.normal) catch |err|
                    log.warn("cursor release unavailable: {t}", .{err});
            },
            .scroll => |scroll| if (scroll.line_delta[1] != 0)
                self.queueZoom(engine, scroll.line_delta[1]),
            .key => |key| switch (key.action) {
                // A repeat changes only continuous controls. Toggles act once
                // per physical press instead of flickering at the repeat rate.
                .press => self.applyKey(engine, key.physical),
                .repeat => switch (key.physical) {
                    .equal, .minus => self.applyKey(engine, key.physical),
                    else => {},
                },
                .release => {},
            },
            else => {},
        }
    }

    fn applyKey(self: *Driver, engine: *lenore.Engine, key: platform.PhysicalKey) void {
        switch (key) {
            .digit_1, .digit_2, .digit_3, .digit_4 => {
                self.tier = @intFromEnum(key) - @intFromEnum(platform.PhysicalKey.digit_1);
                const chosen = tiers[self.tier];
                log.info("tier {s}: {d} steps at {d:.3} rad, {d:.1} windings of budget", .{
                    chosen.name, chosen.max_steps, chosen.max_turn, chosen.windings(),
                });
            },
            .equal => self.setExposure(self.exposure * 1.15),
            .minus => self.setExposure(self.exposure / 1.15),
            .t => {
                self.thickness = if (self.thickness > 0.09) 0.02 else self.thickness * 1.6;
                log.info("disk half thickness {d:.3} of the radius", .{self.thickness});
            },
            .o => {
                self.opacity = if (self.opacity > 4.0) 0.0 else @max(self.opacity * 2.0, 0.15);
                log.info("disk opacity {d:.2}", .{self.opacity});
            },
            .d => {
                self.debug = (self.debug + 1) % 4;
                log.info("debug {d}: {s}", .{ self.debug, debugNames[self.debug] });
            },
            .r => {
                self.spin = -self.spin;
                log.info("disk reversed: spin {d}", .{self.spin});
            },
            .c => {
                self.doppler_colour = !self.doppler_colour;
                log.info("doppler colour {s}", .{if (self.doppler_colour) "on" else "off"});
            },
            .escape => engine.requestExit(),
            else => {},
        }
    }

    fn setExposure(self: *Driver, value: f32) void {
        self.exposure = std.math.clamp(value, 0.05, 64);
        log.info("exposure {d:.2}: the disk's peak against the operator's white", .{self.exposure});
    }

    fn queueZoom(self: *Driver, engine: *lenore.Engine, wheel_delta: f32) void {
        if (!std.math.isFinite(wheel_delta)) return;

        switch (engine.camera.projection) {
            .perspective => |perspective| {
                const current = @tan(perspective.fov_y * 0.5);
                const requested = self.zoom_target orelse current;
                const factor = std.math.pow(f32, zoom_step, wheel_delta);
                self.zoom_target = std.math.clamp(
                    requested * factor,
                    @tan(min_fov_y * 0.5),
                    @tan(max_fov_y * 0.5),
                );
            },
        }
    }

    fn advanceZoom(self: *Driver, engine: *lenore.Engine, delta: f32) void {
        const target = self.zoom_target orelse return;

        switch (engine.camera.projection) {
            .perspective => |*perspective| {
                const current = @tan(perspective.fov_y * 0.5);
                const response = 1.0 - @exp(-zoom_response * delta);
                var next = current + (target - current) * response;
                if (@abs(next - target) <= zoom_snap * target) {
                    next = target;
                    self.zoom_target = null;
                }
                perspective.fov_y = 2.0 * std.math.atan(next);
            },
        }
    }

    pub fn onResize(_: *Driver, _: *lenore.Engine, _: platform.Extent2D) !void {}
    pub fn onCompute(_: *Driver, _: *lenore.Engine, _: *lenore.Level, _: gpu.vk.CommandBuffer) !void {}

    // Nothing to draw over the picture. The hook is required of every

    // driver, so declining it is a declaration rather than an omission.
    pub fn onUiRegions(_: *Driver, _: *lenore.Engine, _: *lenore.Level, _: *imui.WidgetContext) !void {}
    pub fn onUiDraw(_: *Driver, _: *lenore.Engine, _: *lenore.Level, _: *imui.WidgetContext) !void {}

    pub fn onRecord(
        self: *Driver,
        engine: *lenore.Engine,
        _: *lenore.Level,
        commands: gpu.vk.CommandBuffer,
    ) !void {
        const ratio = engine.aspect();
        const basis = try engine.camera.rayBasis(ratio);
        const eye = engine.camera.placement().position;
        const chosen = tiers[self.tier];

        // Through the same conversion the engine's own camera block goes
        // through, and not around it.
        //
        // The scene's basis is in world orientation, where the vertical axis
        // points up. A shader building `front + right * x + up * y` indexes it
        // by a device coordinate, and Vulkan's device y points down, so the two
        // disagree by a sign. `vulkanClipCamera` is where that sign lives; a
        // shader taking the basis directly draws the world inverted about the
        // horizon, which on this subject looks like the near half of the disk
        // being cut away and the shadow standing in front of it.
        const framebuffer = gpu.vulkanClipCamera(.{
            .view_projection = zm.identity(),
            .position = .{ eye[0], eye[1], eye[2], 1 },
            .ray_right = lane(basis.right),
            .ray_up = lane(basis.up),
            .ray_front = lane(basis.front),
        }).block;

        self.effect.record(commands, .{
            .ray_right = framebuffer.ray_right,
            .ray_up = framebuffer.ray_up,
            .ray_front = framebuffer.ray_front,
            .eye = .{ eye[0], eye[1], eye[2], 1 },
            .max_steps = chosen.max_steps,
            .max_turn = chosen.max_turn,
            .disk_inner = disk_inner,
            .disk_outer = disk_outer,
            .disk_temperature = disk_temperature,
            .disk_spin = self.spin,
            .doppler_colour = @intFromBool(self.doppler_colour),
            .exposure = self.exposure,
            .star_intensity = self.star_intensity,
            .time = self.elapsed,
            .turbulence = self.turbulence,
            .disk_thickness = self.thickness,
            .disk_opacity = self.opacity,
            .debug = self.debug,
        });
    }

    pub fn onUpdate(
        self: *Driver,
        engine: *lenore.Engine,
        _: *lenore.Level,
        time: lenore.FrameTime,
    ) !void {
        // The clamped delta and not the wall interval: neither the disk nor a
        // camera throw may teleport after a stall.
        self.elapsed += time.delta;
        self.orbit.advance(&engine.camera, time.delta);
        self.advanceZoom(engine, time.delta);

        // Reported per closed window rather than per frame, and only when the
        // window is one this has not seen: the report's own numbers are not an
        // identity, since two windows can measure the same duration.
        if (engine.metrics.closed_windows == self.reported_windows) return;
        self.reported_windows = engine.metrics.closed_windows;

        const report = engine.metrics.last_fps orelse return;
        // The chain's own counter, not the setting: what a look asked for and
        // what was recorded are two different things, and this is the one that
        // can be read from outside a frame. A glow that is not there is either
        // a chain that never ran or a disk under the threshold, and these two
        // numbers separate them.
        log.info("{s}: {d:.1} fps, exposure {d:.2}, {d} bloom chains over {d} levels", .{
            tiers[self.tier].name,
            report.fps,
            self.exposure,
            engine.renderer.bloomChains(),
            engine.renderer.bloomLevels(),
        });
    }
};

// A direction in the four lanes the shader's vector field has. The last one is
// the alignment's padding and is read by nothing.
fn lane(direction: [3]f32) [4]f32 {
    return .{ direction[0], direction[1], direction[2], 0 };
}

fn tierByName(name: []const u8) ?usize {
    for (tiers, 0..) |tier, index| {
        if (std.mem.eql(u8, tier.name, name)) return index;
    }
    return null;
}

const checking = std.debug.runtime_safety;
var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

pub fn main(process: std.process.Init.Minimal) !void {
    const gpa = if (checking) debug_allocator.allocator() else std.heap.smp_allocator;
    defer if (checking) {
        _ = debug_allocator.deinit();
    };

    var tier: usize = default_tier;
    var spin: f32 = 1;
    {
        // initAllocator because on Windows there is no other. Nothing outlives
        // the block: what the parse keeps is a tier index and a sign.
        var iterator: std.process.Args.Iterator = try .initAllocator(process.args, gpa);
        defer iterator.deinit();
        _ = iterator.skip();
        while (iterator.next()) |argument| {
            if (std.mem.startsWith(u8, argument, "--tier=")) {
                tier = tierByName(argument["--tier=".len..]) orelse {
                    log.err("{s}", .{usage});
                    return error.UnknownTier;
                };
            } else if (std.mem.eql(u8, argument, "--spin=-1")) {
                spin = -1;
            } else if (std.mem.eql(u8, argument, "--spin=1")) {
                spin = 1;
            } else {
                log.err("{s}", .{usage});
                return error.UnknownArgument;
            }
        }
    }

    var engine: lenore.Engine = undefined;
    try engine.init(gpa, .{
        .title = "Lenore: Schwarzschild",
        .extent = .{ .width = 1280, .height = 720 },
        // Nothing is drawn from an asset, and zero is how that is said.
        .frame_capacity = .{ .instances = 0, .joints = 0 },
        .morph_capacity = .{ .meshes = 0, .weights = 0 },
        .material_capacity = 0,
    });
    defer engine.deinit();

    engine.camera.anchor = .{ .orbit = .{ .target = .{ 0, 0, 0 }, .distance = default_distance } };
    engine.camera.pitch = default_pitch;

    // The disk carries its brightness in the target's own units and the chain
    // is what spreads it, so bloom is not decoration here: without it the
    // beamed edge is a hard white line rather than something that glows.
    engine.look.bloom = .{};

    var level: lenore.Level = .empty(gpa);
    defer level.deinit(&engine.textures);

    var effect = try Effect.init(&engine.context, engine.renderer.mainPassFormats());
    // Before the engine's own teardown, and after the drain inside it would be
    // too late: these are destroyed while the device may still be running the
    // last frame that named them.
    defer {
        engine.context.waitIdle() catch {};
        effect.deinit();
    }

    var driver: Driver = .{ .effect = &effect, .tier = tier, .spin = spin };
    log.info("{s}", .{usage});
    log.info("tier {s}: {d} steps, {d:.1} windings of budget", .{
        tiers[tier].name,
        tiers[tier].max_steps,
        tiers[tier].windings(),
    });

    try engine.run(&level, &driver);
}
