// Inertial mouse orbit shared by examples that keep the engine camera anchored
// to a subject.
//
// This owns input interpretation, not camera state: yaw, pitch and the orbit
// radius remain in `scene.Camera`. A caller owns pointer capture because losing
// focus and a backend refusing capture are window policy, not camera motion.

const std = @import("std");
const scene = @import("lenore-scene");

pub const OrbitControl = struct {
    dragging: bool = false,
    last_cursor: [2]f32 = .{ 0, 0 },
    last_timestamp_ns: u64 = 0,

    // Yaw and pitch in radians per second. This survives release and decays in
    // `advance`; taking hold or cancelling discards it explicitly.
    angular_velocity: [2]f32 = .{ 0, 0 },

    pub const sensitivity: f32 = 0.0025;
    pub const velocity_response: f32 = 0.035;
    pub const damping: f32 = 4.0;
    pub const max_angular_speed: f32 = 8.0;
    pub const stop_speed: f32 = 0.002;

    pub fn begin(self: *OrbitControl, position: [2]f32, timestamp_ns: u64) void {
        self.dragging = true;
        self.last_cursor = position;
        self.last_timestamp_ns = timestamp_ns;
        // A new gesture takes ownership from the previous throw rather than
        // adding two unrelated impulses.
        self.angular_velocity = .{ 0, 0 };
    }

    pub fn dragTo(
        self: *OrbitControl,
        camera: *scene.Camera,
        position: [2]f32,
        timestamp_ns: u64,
    ) void {
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
                self.angular_velocity[axis] +=
                    (instantaneous - self.angular_velocity[axis]) * response;
            }
        }

        self.last_cursor = position;
        self.last_timestamp_ns = timestamp_ns;
    }

    // Apply the atomic button position before releasing. A coalesced cursor
    // event may be older than the button callback, and dropping that final
    // delta makes the throw depend on event batching.
    pub fn end(
        self: *OrbitControl,
        camera: *scene.Camera,
        position: [2]f32,
        timestamp_ns: u64,
    ) void {
        self.dragTo(camera, position, timestamp_ns);
        self.dragging = false;
    }

    // Focus loss is not a throw. It can arrive without a release event, so both
    // ownership and velocity are cleared.
    pub fn cancel(self: *OrbitControl) void {
        self.dragging = false;
        self.angular_velocity = .{ 0, 0 };
    }

    pub fn advance(self: *OrbitControl, camera: *scene.Camera, delta: f32) void {
        if (self.dragging) return;

        if (@abs(self.angular_velocity[0]) < stop_speed and
            @abs(self.angular_velocity[1]) < stop_speed)
        {
            self.angular_velocity = .{ 0, 0 };
            return;
        }

        // Integrate exponential decay analytically. `velocity * delta` followed
        // by attenuation travels farther at a low frame rate; this integral
        // gives one frame and any subdivision of it the same angle.
        const attenuation = @exp(-damping * delta);
        const travel = (1.0 - attenuation) / damping;
        camera.yaw = wrapped(camera.yaw + self.angular_velocity[0] * travel);
        camera.pitch = wrapped(camera.pitch + self.angular_velocity[1] * travel);
        self.angular_velocity[0] *= attenuation;
        self.angular_velocity[1] *= attenuation;
    }
};

// Representation is bounded for trigonometric precision, motion is not: crossing
// either endpoint continues from the other with the same orientation and
// derivative, so this imposes no pole or axis clamp.
fn wrapped(angle: f32) f32 {
    return angle - @floor((angle + std.math.pi) / std.math.tau) * std.math.tau;
}

const testing = std.testing;

fn expectApprox(expected: f32, actual: f32) !void {
    try testing.expectApproxEqAbs(expected, actual, 1e-5);
}

test "drag applies both axes without a vertical clamp" {
    var control: OrbitControl = .{};
    var camera: scene.Camera = .{};

    control.begin(.{ 10, 20 }, 1_000_000);
    control.dragTo(&camera, .{ 110, -980 }, 17_000_000);

    try expectApprox(0.25, camera.yaw - (-std.math.pi / 2.0));
    try expectApprox(2.5, camera.pitch);
    try testing.expect(camera.pitch > std.math.pi / 2.0);
}

test "one inertial interval equals the same interval subdivided" {
    var whole: OrbitControl = .{ .angular_velocity = .{ 1.25, -0.75 } };
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

test "begin and focus cancellation stop an existing throw" {
    var control: OrbitControl = .{ .angular_velocity = .{ 3, -2 } };
    control.begin(.{ 0, 0 }, 10);
    try testing.expectEqual([2]f32{ 0, 0 }, control.angular_velocity);
    try testing.expect(control.dragging);

    control.angular_velocity = .{ 1, 1 };
    control.cancel();
    try testing.expectEqual([2]f32{ 0, 0 }, control.angular_velocity);
    try testing.expect(!control.dragging);
}
