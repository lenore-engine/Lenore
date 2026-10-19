const std = @import("std");
const fps = @import("lenore-fps");
const gltf = @import("lenore-gltf");
const gpu = @import("lenore-gpu");
const lenore = @import("lenore");
const platform = @import("lenore-platform");

const range = @import("range.zig");

const log = std.log.scoped(.fps_range);

// A flat range to shoot on: the first consumer of `lenore-fps`. A body walks
// with the gait, a pistol is held, kicked and swayed by the module and drawn
// where it is held, and every round is traced through the range's boxes and
// reported to the log.
//
// Usage: run-fps_range -- <pistol.glb>
//
// The pistol is a test model kept outside the repository. Its origin is by
// the trigger, +Z out of the muzzle, and the muzzle and sight numbers below
// are measured on it.
//
//   left click     take the pointer, then fire
//   right button   raise the sights while held
//   r              reload
//   b              next fire mode
//   w a s d        walk, shift to hurry
//   escape         give the pointer back, then quit

const checking = std.debug.runtime_safety;
var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

const capacity: gpu.FrameCapacity = .{ .instances = 64, .joints = 0 };

// A full-size polymer pistol in 9x19: seventeen in the magazine, the slide
// locks on the last round. The rate is how fast a trigger is pulled, not a
// mechanism's.
const pistol_action: fps.ActionSpec = .{
    .modes = .init(.{ .semi = true }),
    .rate = 360.0,
    .capacity = 17,
    .locks_open = true,
    .reload = 1.8,
    .reload_empty = 2.3,
};

// Magazines carried in all, the one in the pistol included.
const magazines = 4;
const Pistol = fps.Action(magazines);

// Taste, not measurement: these are the first numbers to change by feel.
const pistol_recoil: fps.RecoilProfile = .{
    .rise = 0.035,
    .drift = 0.012,
    .recovery = 0.7,
    .view = .{ .frequency = 4.0, .damping = 0.8 },
    .kick_back = 0.03,
    .muzzle_rise = 0.12,
    .muzzle_drift = 0.02,
    .hand = .{ .frequency = 9.0, .damping = 0.55 },
    .shake = 0.15,
};

const pistol_handling: fps.HandlingProfile = .{
    // Low and right, turned in a little so the barrel crosses the line of
    // sight about ten metres out.
    .hip = .{ .position = .{ 0.13, -0.15, 0.30 }, .yaw = -0.013, .pitch = 0.015 },
    // The model's origin is by the trigger and the sights stand about eight
    // centimetres above it, so this puts the sights on the eye.
    .aimed = .{ .position = .{ 0.0, -0.082, 0.28 } },
    .raise = 0.22,
    .inertia = 0.5,
    .inertia_limit = 0.08,
    .look = .{ .frequency = 3.0, .damping = 0.75 },
    .sway = 0.02,
    .sway_rise = 9.0,
    .sway_settle = 5.5,
    .bob = 0.5,
    .drift_angle = 0.004,
    .drift_shift = 0.003,
    .steady = 0.25,
    // The front face of the barrel in the exported model's glTF axes.
    .muzzle = .{ 0.0, 0.061, 0.101 },
};

// A 9x19 ball round in the range's units: through a partition and a torso,
// not through the plate.
const ball: fps.Round = .{ .damage = 100.0, .penetration = 10.0, .range = 100.0 };
// Radians: the cone the pistol and the round put a shot inside, about 7 cm at
// 25 m.
const spread: f32 = 0.0015;

const walking: fps.WalkTuning = .{
    .stride = 1.5,
    .vert = 0.009,
    .lat = 0.006,
    .roll = 0.0035,
    .pitch = 0.0015,
    .sharp = 0.35,
    .jitter = 0.12,
    .limp = 0.05,
    .scale = 1.0,
};
const eye_height: f32 = 1.68;
// Metres a second. A careful walk, and what holding shift makes of it.
const walk_speed: f32 = 1.8;
const hurry: f32 = 1.9;
// What one footfall adds to the shake.
const footfall_kick: f32 = 0.05;

// The pistol's index among the level's props.
const pistol_prop = 0;

const Range = struct {
    pub const hooks: lenore.Hooks(Range) = .{ .event = event, .update = update };

    look: lenore.MouseLook = .{},
    keys: lenore.MoveKeys = .{ .bindings = .{
        .up = &.{},
        .down = &.{},
        .fast = &.{.shift_left},
    } },
    captured: bool = false,

    // Where the player put the feet and the aim. The camera is these plus
    // what the body and the weapon do to them, and is never read back.
    feet: [2]f32 = range.spawn,
    yaw: f32 = -std.math.pi / 2.0,
    pitch: f32 = 0.0,

    view: fps.Aim = .{ .lag = 0.02, .yaw = -std.math.pi / 2.0 },
    walk: fps.Walk = .{ .tuning = walking },
    lean: fps.Lean = .{ .gain = 0.03 / walk_speed },
    shake: fps.Shake = .{},
    hand: fps.Handheld = .{ .angle = 0.002, .shift = 0.002 },
    walked: f32 = 0.0,

    action: Pistol,
    recoil: fps.Recoil,
    handling: fps.Handling,
    // Shots fired, which keys each one's scatter.
    fired: u32 = 0,

    fn event(self: *Range, engine: *lenore.Engine, input: platform.Event, ui_consumed: bool) !void {
        if (ui_consumed) return;
        switch (input.payload) {
            .cursor => |cursor| {
                const turned = self.look.turn(cursor.logical_position, .{ .yaw = self.yaw, .pitch = self.pitch });
                self.yaw = turned.yaw;
                self.pitch = turned.pitch;
            },
            .mouse_button => |button| switch (button.button) {
                .left => switch (button.action) {
                    .press => if (self.captured) self.action.pull() else self.capture(engine, button.logical_position),
                    .release => self.action.release(),
                    .repeat => {},
                },
                .right => self.handling.aiming = button.action != .release,
                else => {},
            },
            .focus => |focus| if (!focus.focused) {
                self.keys.cancel();
                self.action.release();
                self.handling.aiming = false;
                self.free(engine);
            },
            .key => |key| switch (key.action) {
                .press => {
                    switch (key.physical) {
                        .escape => return if (self.captured) self.free(engine) else engine.requestExit(),
                        .r => self.action.reload() catch |err| log.info("no reload: {t}", .{err}),
                        .b => log.info("mode: {t}", .{self.action.cycleMode()}),
                        else => {},
                    }
                    _ = self.keys.key(key.physical, .press);
                },
                .release => _ = self.keys.key(key.physical, .release),
                .repeat => {},
            },
            else => {},
        }
    }

    // The pointer is the view's while captured: hidden, unbounded, and seeded
    // where it was clicked so the first motion is not a jump.
    fn capture(self: *Range, engine: *lenore.Engine, cursor: [2]f32) void {
        engine.window.setCursorMode(.disabled) catch |err|
            log.warn("cursor capture unavailable: {t}", .{err});
        self.look.begin(cursor);
        self.captured = true;
    }

    fn free(self: *Range, engine: *lenore.Engine) void {
        if (!self.captured) return;
        engine.window.setCursorMode(.normal) catch |err|
            log.warn("cursor release unavailable: {t}", .{err});
        self.look.end();
        self.captured = false;
    }

    fn update(self: *Range, engine: *lenore.Engine, level: *lenore.Level, time: lenore.FrameTime) !void {
        const delta = time.delta;
        const before_yaw = self.view.yaw;
        const before_pitch = self.view.pitch;
        self.view.follow(self.yaw, self.pitch, delta);

        // The walk, along the view's heading and level whatever the pitch.
        const forward: [2]f32 = .{ @cos(self.view.yaw), @sin(self.view.yaw) };
        const right: [2]f32 = .{ -@sin(self.view.yaw), @cos(self.view.yaw) };
        const axes = self.keys.axes();
        var wish: [2]f32 = .{
            forward[0] * axes.forward + right[0] * axes.strafe,
            forward[1] * axes.forward + right[1] * axes.strafe,
        };
        const length = @sqrt(wish[0] * wish[0] + wish[1] * wish[1]);
        if (length > 0.0) {
            const pace = walk_speed * (if (self.keys.held(.fast)) hurry else 1.0) * delta / length;
            wish = .{ wish[0] * pace, wish[1] * pace };
        }
        const before = self.feet;
        self.feet = range.slide(self.feet, wish);
        const moved: [2]f32 = .{ self.feet[0] - before[0], self.feet[1] - before[1] };
        const velocity: [2]f32 = if (delta > 0.0) .{ moved[0] / delta, moved[1] / delta } else .{ 0.0, 0.0 };
        const across = velocity[0] * right[0] + velocity[1] * right[1];
        const along = velocity[0] * forward[0] + velocity[1] * forward[1];

        // What carrying the eye does, as Backrooms composes it.
        self.walked += delta;
        const sway = self.walk.update(@sqrt(along * along + across * across), delta, self.walked);
        if (sway.footfall) self.shake.add(footfall_kick);
        const knock = self.shake.update(delta);
        const drift = self.hand.update(delta);
        const tip = self.lean.update(across, delta);

        // The weapon, then the shots it fires this frame.
        self.handling.update(.{
            .turn_yaw = self.view.yaw - before_yaw,
            .turn_pitch = self.view.pitch - before_pitch,
            .right = across,
            .forward = along,
            .sway = sway,
        }, delta);
        self.recoil.update(delta);

        const kicked = self.recoil.view();
        engine.camera.anchor = .{ .eye = .{
            self.feet[0] + right[0] * (sway.right + drift.right),
            eye_height + sway.up + drift.up,
            self.feet[1] + right[1] * (sway.right + drift.right),
        } };
        engine.camera.yaw = self.view.yaw + knock.yaw + drift.yaw + kicked.yaw;
        engine.camera.pitch = self.view.pitch + sway.pitch + knock.pitch + drift.pitch + kicked.pitch;
        engine.camera.roll = sway.roll + knock.roll + drift.roll + tip;

        // Placed against the camera just set, so the pistol is drawn in this
        // frame's view and not the last one's.
        const held = self.handling.place(self.handling.pose(self.recoil.weapon()), engine.camera.placement());
        try level.world.placeProp(pistol_prop, 0, held.model);

        // The clamped delta in nanoseconds, so a stall does not owe a volley.
        const report = self.action.advance(@min(time.interval_ns, lenore.max_frame_delta_ns));
        for (0..report.shots) |_| self.shoot(engine);
        if (report.dry) log.info("click", .{});
        if (report.locked) log.info("slide locked back", .{});
        if (report.reloaded) log.info("reloaded: {d} loaded, spares {any}", .{ self.action.loaded(), self.action.spareMagazines() });
    }

    // One round: from the muzzle along the barrel as it is held this frame,
    // before the round's own kick moves it.
    fn shoot(self: *Range, engine: *lenore.Engine) void {
        const held = self.handling.place(self.handling.pose(self.recoil.weapon()), engine.camera.placement());
        const direction = fps.scatter(held.direction, spread, 0x5eed, self.fired);
        self.fired +%= 1;

        var crossings: [range.props.len]fps.Crossing = undefined;
        const count = range.cross(.{ .origin = held.muzzle, .direction = direction }, &crossings);
        var hits: [range.props.len]fps.Hit = undefined;
        const outcome = fps.resolve(ball, crossings[0..count], &hits);

        log.info("shot {d}, {d} left", .{ self.fired, self.action.loaded() });
        for (hits[0..outcome.hits]) |hit| {
            const prop = range.props[hit.tag];
            log.info("  {s} at {d:.1} m: arrived with {d:.0}%, {d:.0} damage{s}", .{
                prop.name,
                hit.distance,
                hit.arrived * 100.0,
                ball.damage * hit.arrived * prop.zone.multiplier(),
                if (hit.stopped) ", stopped" else "",
            });
        }

        const kick = self.recoil.kick();
        self.yaw += kick.climb_yaw;
        self.pitch = std.math.clamp(self.pitch + kick.climb_pitch, -self.look.settings.max_pitch, self.look.settings.max_pitch);
        self.shake.add(kick.shake);
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

    var arguments: std.process.Args.Iterator = try .initAllocator(process.args, gpa);
    defer arguments.deinit();
    _ = arguments.next();
    const pistol_path = arguments.next() orelse {
        log.err("usage: run-fps_range -- <pistol.glb>", .{});
        return error.BadArguments;
    };

    var root = try std.Io.Dir.cwd().openDir(io, std.fs.path.dirname(pistol_path) orelse ".", .{});
    defer root.close(io);
    var loaded = try gltf.loader.open(gpa, io, root, std.fs.path.basename(pistol_path));
    defer loaded.deinit(gpa);
    var pistol = try gltf.importer.build(gpa, &loaded.document, loaded.directory);
    defer pistol.deinit(gpa);
    try gltf.importer.readExternalImages(gpa, io, root, &pistol);

    var model = try range.build(gpa);
    defer model.deinit(gpa);

    var engine: lenore.Engine = undefined;
    try engine.init(gpa, .{
        .title = "Lenore: fps range",
        .frame_capacity = capacity,
        .gpu_timing = true,
        // The level's materials and the pistol's share one table.
        .material_capacity = @intCast(model.materials.len + pistol.materials.len),
    });
    defer engine.deinit();

    const lights = [_]gpu.LightUniform{
        lenore.packLight(try .directional(.{ 1, 1, 1 }, std.math.pi, .{ 0.4, -0.8, -0.3 })),
    };
    engine.look = .{
        .lights = &lights,
        .sun_shadow = .{ .enabled = true, .strength = 1, .normal_offset_texels = 1 },
    };

    var level = try lenore.Level.init(gpa, engine.deps(), &model, .{
        .capacity = capacity,
        .props = &.{.{ .model = &pistol, .name = "pistol" }},
    });
    defer level.deinit(&engine.textures);
    try engine.install(&level, &model);

    // After `install`, which frames the whole range and sets the near plane
    // from its size: a metre or so, which would cut away a pistol held 30 cm
    // from the eye. Sixty degrees up and down is a wide first-person view.
    engine.camera.projection = .{ .perspective = .{ .fov_y = std.math.pi / 3.0, .near = 0.01, .far = 200.0 } };

    var shooter: Range = .{
        .action = try Pistol.init(pistol_action, magazines - 1),
        .recoil = try .init(pistol_recoil),
        .handling = try .init(pistol_handling),
    };
    var report: lenore.FrameLog = .{};
    try engine.run(&level, .{ &shooter, &report });
}
