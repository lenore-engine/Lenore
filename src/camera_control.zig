// Turning and moving a view from the pointer and the keyboard.
//
// Two layers. `MouseLook` and `MoveKeys` are arithmetic over input: they hold
// what a gesture needs between events and answer with a value, and where that
// value lands is the caller's. A walk controller with collisions builds on
// them. `FlyCamera` and `OrbitCamera` are finished components over
// `scene.Camera`, added to `Engine.run`'s tuple by an application that wants
// one of the two common cameras and nothing else.
//
// The camera's pose stays in `scene.Camera` throughout. Nothing here keeps a
// second copy of yaw, pitch or position.

const std = @import("std");
const platform = @import("lenore-platform");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const engine_module = @import("engine.zig");
const Engine = engine_module.Engine;
const Hooks = engine_module.Hooks;
const Level = @import("level.zig").Level;
const FrameTime = @import("time.zig").FrameTime;

const log = std.log.scoped(.lenore);

// A view turned by the pointer: a cursor position and an aim in, the aim after
// the pointer moved out.
//
// Returned rather than written, because the aim is not always the camera's.
// A walking body sways the camera every frame, and turning from where the
// camera ended up would fold the sway into the aim; such a caller keeps its
// own aim and puts the camera on it. `turnCamera` is the common case.
pub const MouseLook = struct {
    pub const Aim = struct {
        yaw: f32,
        pitch: f32,
    };

    pub const Settings = struct {
        // Radians of turn per logical unit the pointer travels.
        sensitivity: f32 = 0.0025,
        // How far the aim tilts up or down. Short of a right angle: past it
        // the view tips over the top and the picture turns upside down.
        max_pitch: f32 = std.math.pi / 2.0 - 0.01,
    };

    settings: Settings = .{},
    looking: bool = false,
    last_cursor: [2]f32 = .{ 0, 0 },

    // Starts a gesture at `cursor`, so the first motion turns by how far the
    // pointer went from here rather than from wherever it last was.
    pub fn begin(self: *MouseLook, cursor: [2]f32) void {
        self.looking = true;
        self.last_cursor = cursor;
    }

    // Ends one. Focus loss calls this too: a release can be lost with focus.
    pub fn end(self: *MouseLook) void {
        self.looking = false;
    }

    pub fn turn(self: *MouseLook, cursor: [2]f32, aim: Aim) Aim {
        if (!self.looking) return aim;
        const moved: [2]f32 = .{ cursor[0] - self.last_cursor[0], cursor[1] - self.last_cursor[1] };
        self.last_cursor = cursor;
        return .{
            .yaw = aim.yaw + moved[0] * self.settings.sensitivity,
            .pitch = std.math.clamp(
                aim.pitch - moved[1] * self.settings.sensitivity,
                -self.settings.max_pitch,
                self.settings.max_pitch,
            ),
        };
    }

    pub fn turnCamera(self: *MouseLook, cursor: [2]f32, camera: *scene.Camera) void {
        const aim = self.turn(cursor, .{ .yaw = camera.yaw, .pitch = camera.pitch });
        camera.yaw = aim.yaw;
        camera.pitch = aim.pitch;
    }
};

// Which way the keyboard asks to move, as six held directions and a modifier.
//
// Held state rather than motion per event: a key repeat arrives at the
// system's repeat rate, so motion driven by events would travel at whatever
// that is set to.
//
// Held per key rather than per direction. Two keys may name one direction,
// and with a flag per direction releasing either would stop it while the other
// was still down. Opposed directions cancel in `axes`.
pub const MoveKeys = struct {
    pub const Direction = enum { forward, back, left, right, up, down, fast };

    pub const Bindings = struct {
        forward: []const platform.PhysicalKey = &.{.w},
        back: []const platform.PhysicalKey = &.{.s},
        left: []const platform.PhysicalKey = &.{.a},
        right: []const platform.PhysicalKey = &.{.d},
        up: []const platform.PhysicalKey = &.{ .space, .e },
        down: []const platform.PhysicalKey = &.{ .shift_left, .shift_right, .q },
        fast: []const platform.PhysicalKey = &.{},
    };

    // Each axis in [-1, 1]: right, world up, and forward positive.
    pub const Axes = struct {
        strafe: f32,
        rise: f32,
        forward: f32,

        pub fn isZero(self: Axes) bool {
            return self.strafe == 0 and self.rise == 0 and self.forward == 0;
        }
    };

    bindings: Bindings = .{},
    pressed: std.EnumSet(platform.PhysicalKey) = .empty,

    // Folds one key event in. Answers whether the key is bound, so a caller can
    // leave the ones it does not own to someone else.
    pub fn key(self: *MoveKeys, physical: platform.PhysicalKey, action: platform.KeyAction) bool {
        if (!self.binds(physical)) return false;
        switch (action) {
            .press, .repeat => self.pressed.insert(physical),
            .release => self.pressed.remove(physical),
        }
        return true;
    }

    // Everything released. Focus can be lost without a key ever coming back up,
    // and a key held at that moment would otherwise stay down for the run.
    pub fn cancel(self: *MoveKeys) void {
        self.pressed = .empty;
    }

    pub fn held(self: *const MoveKeys, direction: Direction) bool {
        const keys = switch (direction) {
            inline else => |tag| @field(self.bindings, @tagName(tag)),
        };
        for (keys) |bound| {
            if (self.pressed.contains(bound)) return true;
        }
        return false;
    }

    pub fn axes(self: *const MoveKeys) Axes {
        return .{
            .strafe = signed(self.held(.right), self.held(.left)),
            .rise = signed(self.held(.up), self.held(.down)),
            .forward = signed(self.held(.forward), self.held(.back)),
        };
    }

    fn binds(self: *const MoveKeys, physical: platform.PhysicalKey) bool {
        inline for (@typeInfo(Bindings).@"struct".field_names) |name| {
            for (@field(self.bindings, name)) |bound| {
                if (bound == physical) return true;
            }
        }
        return false;
    }

    fn signed(positive: bool, negative: bool) f32 {
        return @as(f32, @floatFromInt(@intFromBool(positive))) - @as(f32, @floatFromInt(@intFromBool(negative)));
    }
};

// A camera that flies free: the pointer turns it, the keys move it along its
// own heading and along world up, and the wheel sets the pace.
//
// Nothing collides, on purpose: a free camera is for looking at whatever is
// there, inside geometry included.
pub const FlyCamera = struct {
    pub const hooks: Hooks(FlyCamera) = .{ .event = event, .update = update };

    pub const Gesture = enum {
        // The left button held turns the view, with the pointer captured for
        // the length of the drag.
        drag,
        // The application decides when the view turns, through `look.begin`
        // and `look.end`; capture and release are its own.
        application,
    };

    look: MouseLook = .{},
    keys: MoveKeys = .{},
    gesture: Gesture = .drag,
    // Off, the component hears nothing and moves nothing, which is how an
    // application with a second mode, a walking body, hands the camera over.
    enabled: bool = true,
    // World units a second at a pace of one.
    speed: f32 = 1.0,
    // The wheel's multiplier on `speed`. Geometric per notch: the useful range
    // spans decades, and a linear step is either imperceptible at the bottom of
    // it or unusable at the top.
    pace: f32 = 1.0,

    const pace_step: f32 = 1.25;
    const min_pace: f32 = 1.0 / 64.0;
    const max_pace: f32 = 64.0;

    fn event(self: *FlyCamera, engine: *Engine, input: platform.Event, ui_consumed: bool) !void {
        if (!self.enabled) return;
        switch (input.payload) {
            // A release is taken even when the UI took it: a key that went down
            // before a widget had focus would otherwise stay down.
            .key => |key| if (!ui_consumed or key.action == .release) {
                _ = self.keys.key(key.physical, key.action);
            },
            .cursor => |cursor| self.turn(&engine.camera, cursor.logical_position),
            .mouse_button => |button| if (self.gesture == .drag and button.button == .left) switch (button.action) {
                .press => if (!ui_consumed) {
                    self.look.begin(button.logical_position);
                    capture(engine, .disabled);
                },
                .release => if (self.look.looking) {
                    self.turn(&engine.camera, button.logical_position);
                    self.look.end();
                    capture(engine, .normal);
                },
                .repeat => {},
            },
            .scroll => |scroll| if (!ui_consumed) self.wheel(scroll.line_delta[1]),
            .focus => |focus| if (!focus.focused) {
                self.keys.cancel();
                if (self.look.looking and self.gesture == .drag) capture(engine, .normal);
                self.look.end();
            },
            else => {},
        }
    }

    fn update(self: *FlyCamera, engine: *Engine, _: *Level, time: FrameTime) !void {
        if (self.enabled) self.advance(&engine.camera, time.delta);
    }

    // One frame of flight, `delta` seconds long.
    pub fn advance(self: *const FlyCamera, camera: *scene.Camera, delta: f32) void {
        const axes = self.keys.axes();
        // No input leaves the camera alone. Moving by nothing would still
        // detach an orbit below, which the next frame does not undo.
        if (axes.isZero()) return;

        const placement = camera.placement();
        // World up rather than the camera's, so rising while looking down still
        // rises: a free camera whose lift followed its pitch cannot fly level.
        const world_up: res.Vec3 = .{ 0, 1, 0 };
        const motion = placement.front * @as(res.Vec3, @splat(axes.forward)) +
            placement.right * @as(res.Vec3, @splat(axes.strafe)) +
            world_up * @as(res.Vec3, @splat(axes.rise));
        // Normalised, so two keys together do not travel root two times as far
        // as one. Opposed keys already cancelled in `axes`; the sum can still be
        // zero when forward and down meet a view pointing straight up, and that
        // is the division this guards.
        const length = @sqrt(@reduce(.Add, motion * motion));
        if (length == 0) return;
        const step = motion * @as(res.Vec3, @splat(self.speed * self.pace * delta / length));

        // Detached first: an orbit places the eye from its pivot, so an eye
        // written into one is ignored.
        camera.detach();
        camera.anchor = .{ .eye = camera.placement().position + step };
    }

    fn turn(self: *FlyCamera, camera: *scene.Camera, cursor: [2]f32) void {
        if (!self.look.looking) return;
        camera.detach();
        self.look.turnCamera(cursor, camera);
    }

    // `lines` arrives already summed when the queue coalesces a burst, so
    // raising the step to it makes one flick worth the notches it contained.
    fn wheel(self: *FlyCamera, lines: f32) void {
        if (lines == 0) return;
        self.pace = std.math.clamp(self.pace * std.math.pow(f32, pace_step, lines), min_pace, max_pace);
    }
};

// A camera turned around what it looks at, with the throw kept after release
// and decaying.
//
// Yaw and pitch stay in `scene.Camera` and the anchor is the caller's to set:
// this turns an orbit and does not choose its pivot or its distance.
pub const OrbitCamera = struct {
    pub const hooks: Hooks(OrbitCamera) = .{ .event = event, .update = update };

    enabled: bool = true,
    // Hide and hold the pointer for the length of a drag. Off where the pointer
    // has to stay visible over an interface while the scene turns.
    capture: bool = true,
    dragging: bool = false,
    last_cursor: [2]f32 = .{ 0, 0 },
    last_timestamp_ns: u64 = 0,
    // Yaw and pitch in radians per second. Kept through a release and decayed
    // in `advance`; taking hold or cancelling discards it.
    angular_velocity: [2]f32 = .{ 0, 0 },

    pub const sensitivity: f32 = 0.0025;
    pub const velocity_response: f32 = 0.035;
    pub const damping: f32 = 4.0;
    pub const max_angular_speed: f32 = 8.0;
    pub const stop_speed: f32 = 0.002;

    fn event(self: *OrbitCamera, engine: *Engine, input: platform.Event, ui_consumed: bool) !void {
        if (!self.enabled) return;
        switch (input.payload) {
            // Not gated on the interface. A drag that began on the scene keeps
            // going: once it holds the pointer no region is under it.
            .cursor => |cursor| self.dragTo(&engine.camera, cursor.logical_position, input.timestamp_ns),
            .mouse_button => |button| if (button.button == .left) switch (button.action) {
                // A press the interface took stops a throw as well as not
                // starting a drag: a click on a panel stops the scene turning.
                .press => if (ui_consumed) self.cancel() else {
                    self.begin(button.logical_position, input.timestamp_ns);
                    if (self.capture) capture(engine, .disabled);
                },
                .release => if (self.dragging) {
                    self.end(&engine.camera, button.logical_position, input.timestamp_ns);
                    if (self.capture) capture(engine, .normal);
                },
                .repeat => {},
            },
            .focus => |focus| if (!focus.focused) {
                if (self.dragging and self.capture) capture(engine, .normal);
                self.cancel();
            },
            else => {},
        }
    }

    fn update(self: *OrbitCamera, engine: *Engine, _: *Level, time: FrameTime) !void {
        if (self.enabled) self.advance(&engine.camera, time.delta);
    }

    pub fn begin(self: *OrbitCamera, position: [2]f32, timestamp_ns: u64) void {
        self.dragging = true;
        self.last_cursor = position;
        self.last_timestamp_ns = timestamp_ns;
        // A new gesture takes over from the previous throw rather than adding
        // two unrelated impulses.
        self.angular_velocity = .{ 0, 0 };
    }

    pub fn dragTo(self: *OrbitCamera, camera: *scene.Camera, position: [2]f32, timestamp_ns: u64) void {
        if (!self.dragging) return;

        const angular_delta = [2]f32{
            (position[0] - self.last_cursor[0]) * sensitivity,
            -(position[1] - self.last_cursor[1]) * sensitivity,
        };
        camera.yaw = wrapped(camera.yaw + angular_delta[0]);
        camera.pitch = wrapped(camera.pitch + angular_delta[1]);

        if (timestamp_ns > self.last_timestamp_ns) {
            const elapsed_ns = timestamp_ns - self.last_timestamp_ns;
            const seconds = @as(f32, @floatFromInt(elapsed_ns)) * 1e-9;
            const response = 1.0 - @exp(-seconds / velocity_response);
            inline for (0..2) |axis| {
                const instantaneous = std.math.clamp(
                    angular_delta[axis] / seconds,
                    -max_angular_speed,
                    max_angular_speed,
                );
                self.angular_velocity[axis] += (instantaneous - self.angular_velocity[axis]) * response;
            }
        }

        self.last_cursor = position;
        self.last_timestamp_ns = timestamp_ns;
    }

    // The button's own position is applied before releasing. A coalesced
    // cursor event may be older than the button's, and dropping that last
    // delta would make the throw depend on how events were batched.
    pub fn end(self: *OrbitCamera, camera: *scene.Camera, position: [2]f32, timestamp_ns: u64) void {
        self.dragTo(camera, position, timestamp_ns);
        self.dragging = false;
    }

    // Focus loss is not a throw. It can arrive without a release, so both the
    // hold and the velocity are cleared.
    pub fn cancel(self: *OrbitCamera) void {
        self.dragging = false;
        self.angular_velocity = .{ 0, 0 };
    }

    pub fn advance(self: *OrbitCamera, camera: *scene.Camera, delta: f32) void {
        if (self.dragging) return;

        if (@abs(self.angular_velocity[0]) < stop_speed and @abs(self.angular_velocity[1]) < stop_speed) {
            self.angular_velocity = .{ 0, 0 };
            return;
        }

        // The decay integrated in closed form. Stepping `velocity * delta` and
        // then attenuating travels farther at a low frame rate; the integral
        // gives one frame and any subdivision of it the same angle.
        const attenuation = @exp(-damping * delta);
        const travel = (1.0 - attenuation) / damping;
        camera.yaw = wrapped(camera.yaw + self.angular_velocity[0] * travel);
        camera.pitch = wrapped(camera.pitch + self.angular_velocity[1] * travel);
        self.angular_velocity[0] *= attenuation;
        self.angular_velocity[1] *= attenuation;
    }

    // The representation is bounded for trigonometric precision and the motion
    // is not: crossing either end continues from the other with the same
    // orientation and derivative, so an orbit has no pole and no clamp.
    fn wrapped(angle: f32) f32 {
        return angle - @floor((angle + std.math.pi) / std.math.tau) * std.math.tau;
    }
};

// Pointer capture for a drag. A backend that refuses it leaves the drag working
// on an uncaptured pointer, which is worth a line in the log and no more.
fn capture(engine: *Engine, mode: platform.CursorMode) void {
    engine.window.setCursorMode(mode) catch |err|
        log.warn("cursor mode {t} unavailable: {t}", .{ mode, err });
}
