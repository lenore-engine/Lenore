const std = @import("std");
const lenore = @import("lenore");
const scene = @import("lenore-scene");

const testing = std.testing;

fn expectApprox(expected: f32, actual: f32) !void {
    try testing.expectApproxEqAbs(expected, actual, 1e-5);
}

test "a look turns by the pointer's travel and stops short of vertical" {
    var look: lenore.MouseLook = .{ .settings = .{ .sensitivity = 0.01, .max_pitch = 1.0 } };
    look.begin(.{ 100, 100 });

    const aim = look.turn(.{ 110, 80 }, .{ .yaw = 0, .pitch = 0 });
    try expectApprox(0.1, aim.yaw);
    // Up the screen is up the view: y grows downward on the pointer.
    try expectApprox(0.2, aim.pitch);

    const clamped = look.turn(.{ 110, -1000 }, aim);
    try expectApprox(1.0, clamped.pitch);
}

test "a look that has not begun leaves the aim alone" {
    var look: lenore.MouseLook = .{};
    const aim: lenore.MouseLook.Aim = .{ .yaw = 0.5, .pitch = -0.25 };
    try testing.expectEqual(aim, look.turn(.{ 400, 400 }, aim));

    look.begin(.{ 0, 0 });
    look.end();
    try testing.expectEqual(aim, look.turn(.{ 400, 400 }, aim));
}

test "a direction stays held while any of its keys is down" {
    var keys: lenore.MoveKeys = .{};
    try testing.expect(keys.key(.space, .press));
    try testing.expect(keys.key(.e, .press));
    _ = keys.key(.space, .release);
    try testing.expect(keys.held(.up));
    _ = keys.key(.e, .release);
    try testing.expect(!keys.held(.up));
}

test "opposed keys cancel and an unbound key is left to others" {
    var keys: lenore.MoveKeys = .{};
    _ = keys.key(.w, .press);
    _ = keys.key(.s, .press);
    _ = keys.key(.d, .press);
    try testing.expectEqual(lenore.MoveKeys.Axes{ .strafe = 1, .rise = 0, .forward = 0 }, keys.axes());

    try testing.expect(!keys.key(.f, .press));
}

test "cancelling releases every key" {
    var keys: lenore.MoveKeys = .{};
    _ = keys.key(.w, .press);
    _ = keys.key(.q, .repeat);
    keys.cancel();
    try testing.expect(keys.axes().isZero());
}

test "a flight with no key down leaves an orbit anchored" {
    var fly: lenore.FlyCamera = .{};
    var camera: scene.Camera = .{ .anchor = .{ .orbit = .{ .target = .{ 0, 0, 0 }, .distance = 5 } } };
    fly.advance(&camera, 1.0);
    try testing.expect(camera.anchor == .orbit);
}

test "a diagonal flight travels as far as a straight one" {
    var fly: lenore.FlyCamera = .{ .speed = 2 };
    var camera: scene.Camera = .{ .anchor = .{ .eye = .{ 0, 0, 0 } } };
    _ = fly.keys.key(.w, .press);
    _ = fly.keys.key(.d, .press);
    fly.advance(&camera, 0.5);

    const eye = camera.placement().position;
    try expectApprox(1.0, @sqrt(@reduce(.Add, eye * eye)));
}

test "the wheel sets the pace geometrically, within its range" {
    var fly: lenore.FlyCamera = .{ .speed = 1 };
    var camera: scene.Camera = .{ .anchor = .{ .eye = .{ 0, 0, 0 } } };
    _ = fly.keys.key(.w, .press);
    fly.pace = 1.25 * 1.25;
    fly.advance(&camera, 1.0);
    const eye = camera.placement().position;
    try expectApprox(1.5625, @sqrt(@reduce(.Add, eye * eye)));
}

test "an orbit drag applies both axes without a vertical clamp" {
    var orbit: lenore.OrbitCamera = .{};
    var camera: scene.Camera = .{};

    orbit.begin(.{ 10, 20 }, 1_000_000);
    orbit.dragTo(&camera, .{ 110, -980 }, 17_000_000);

    try expectApprox(0.25, camera.yaw - (-std.math.pi / 2.0));
    try expectApprox(2.5, camera.pitch);
    try testing.expect(camera.pitch > std.math.pi / 2.0);
}

test "one inertial interval equals the same interval subdivided" {
    var whole: lenore.OrbitCamera = .{ .angular_velocity = .{ 1.25, -0.75 } };
    var split = whole;
    var whole_camera: scene.Camera = .{ .yaw = 0, .pitch = 0 };
    var split_camera = whole_camera;

    whole.advance(&whole_camera, 0.5);
    split.advance(&split_camera, 0.25);
    split.advance(&split_camera, 0.25);

    try expectApprox(whole_camera.yaw, split_camera.yaw);
    try expectApprox(whole_camera.pitch, split_camera.pitch);
    try expectApprox(whole.angular_velocity[0], split.angular_velocity[0]);
    try expectApprox(whole.angular_velocity[1], split.angular_velocity[1]);
}

test "taking hold and losing focus both stop a throw" {
    var orbit: lenore.OrbitCamera = .{ .angular_velocity = .{ 3, -2 } };
    orbit.begin(.{ 0, 0 }, 10);
    try testing.expectEqual([2]f32{ 0, 0 }, orbit.angular_velocity);
    try testing.expect(orbit.dragging);

    orbit.angular_velocity = .{ 1, 1 };
    orbit.cancel();
    try testing.expectEqual([2]f32{ 0, 0 }, orbit.angular_velocity);
    try testing.expect(!orbit.dragging);
}

// The components' hooks are private functions reached only through `run`, so
// this is what compiles them.
test "both cameras run as components" {
    var fly: lenore.FlyCamera = .{};
    var orbit: lenore.OrbitCamera = .{};
    const engine: *lenore.Engine = undefined;
    const level: *lenore.Level = undefined;
    _ = @TypeOf(engine.run(level, .{ &fly, &orbit }));
}
